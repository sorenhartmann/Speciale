import logging
from typing import Dict, Optional, Sized, Tuple, cast

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from torch import Tensor
from torch.distributions import Gamma

from src.bayesian.core import (
    BayesianConversionConfig,
    iter_bayesian_modules,
    to_bayesian_model,
)
from src.bayesian.priors import NormalPrior
from src.inference.base import BATCH_IN, InferenceModule
from src.inference.mcmc.samplable import ParameterPosterior
from src.inference.mcmc.sample_containers import (
    CompleteSampleContainer,
    SampleContainer,
)
from src.inference.mcmc.samplers import SGHMC, Sampler
from src.inference.mcmc.variance_estimators import NextEpochException, NoStepException
from src.models.base import ErrorRate, Model

log = logging.getLogger(__name__)


class MCMCInference(InferenceModule):
    def __init__(
        self,
        model: Model,
        sampler: Optional[Sampler] = None,
        sample_container: Optional[SampleContainer] = None,
        burn_in: int = 0,
        steps_per_sample: Optional[int] = None,
        use_gibbs_step: bool = True,
        prior_config: Optional[BayesianConversionConfig] = None,
        # filter_samples_before_test: float = 1.0,
    ):

        super().__init__()

        if sampler is None:
            sampler = SGHMC()

        if sample_container is None:
            sample_container = CompleteSampleContainer()

        self.sample_container = sample_container
        self.automatic_optimization = False

        self.model = to_bayesian_model(model, prior_config)
        self.posterior = ParameterPosterior(self.model)
        self.sampler = sampler

        self.burn_in = burn_in
        self.steps_per_sample = steps_per_sample
        self.use_gibbs_step = use_gibbs_step

        self.val_metrics = torch.nn.ModuleDict(self.model.get_metrics())
        self._precision_gibbs_step()

    def configure_optimizers(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        self.sampler.setup(self.posterior)

    def on_fit_start(self) -> None:
        self.val_preds: Dict[int, Dict[int, Tensor]] = {}

    def on_train_start(self) -> None:

        trainer = cast(Trainer, self.trainer)

        if self.steps_per_sample is None:
            self.steps_per_sample = len(trainer.train_dataloader)

        self.step_until_next_sample = self.steps_per_sample
        self.burn_in_remaining = self.burn_in

    def on_train_batch_start(
        self,
        batch: Tuple[Tensor, Tensor],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if getattr(self, "_finish_train_epoch", False):
            self._finish_train_epoch = False
            return -1  # type: ignore
        else:
            return None

    def on_train_epoch_start(self) -> None:
        self.sampler.on_train_epoch_start(self)

    def training_step(self, batch: BATCH_IN, batch_idx: int) -> STEP_OUTPUT:  # type: ignore

        trainer = cast(Trainer, self.trainer)
        train_dataset = cast(
            Sized, trainer.train_dataloader.dataset
        )  # Only sized datasets

        x, y = batch
        sampling_fraction = len(x) / len(train_dataset)

        try:
            with self.posterior.observe(x, y, sampling_fraction):
                self.sampler.next_sample(return_sample=False)
        except NoStepException:
            return None  # type: ignore
        except NextEpochException:
            self._finish_train_epoch = True
            return None  # type: ignore

        # Only procceed after last batch
        self.step_until_next_sample -= 1

        is_last_batch = self.step_until_next_sample == 0
        if not is_last_batch:
            return None  # type: ignore
        else:
            self.step_until_next_sample = cast(int, self.steps_per_sample)

        if self.use_gibbs_step:
            self._precision_gibbs_step()

        # Burn in
        burn_in = self.burn_in_remaining > 0
        if burn_in:
            self.burn_in_remaining -= 1
            return None  # type: ignore

        # Register sample with sample container. Pruning is handled by container
        def get_sample() -> Tensor:
            return self.posterior.state.clone().detach()

        self.sample_container.register_sample(get_sample)

    def _precision_gibbs_step(self) -> None:

        for module in iter_bayesian_modules(self.model):

            for name, prior in module.priors.items():

                if not isinstance(prior, NormalPrior):
                    continue

                parameter = getattr(module, name)
                alpha = 1.0 + parameter.numel() / 2
                beta = 1.0 + parameter.square().sum() / 2
                new_precision = Gamma(alpha, beta).sample()
                prior.precision.copy_(new_precision)

    @torch.no_grad()
    def validation_step(self, batch: BATCH_IN, batch_idx: int) -> Optional[STEP_OUTPUT]:  # type: ignore

        if len(self.sample_container) == 0:
            return None

        x, y = batch

        pred = torch.tensor(0.0)

        for i, sample in self.sample_container.items():

            if i not in self.val_preds:
                self.val_preds[i] = {}

            if batch_idx in self.val_preds[i]:
                pred = pred + self.val_preds[i][batch_idx]

            else:
                old_state = self.sampler.samplable.state
                self.sampler.samplable.state = sample
                output = self.model(x)
                pred_ = self.model.predict_gvn_output(output)
                pred = pred + pred_
                self.val_preds[i][batch_idx] = pred_
                self.sampler.samplable.state = old_state

        pred /= len(self.sample_container)

        for name, metric in self.val_metrics.items():
            self.log(f"{name}/val", metric(pred, y), prog_bar=True)

        return None

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:

        # Delete predictions no longer in use
        delete_keys = set(self.val_preds) - set(self.sample_container.samples)
        for key in delete_keys:
            del self.val_preds[key]
            # del self.val_joint_logliks[key]
            # del self.val_avg_likelihood[key]

    def on_test_epoch_start(self) -> None:

        self.test_metric = ErrorRate().to(device=self.device)
        # if self.filter_samples_before_test != 1:
        #     sample_logits = self.get_sample_logits()

    @torch.no_grad()
    def test_step(self, batch: BATCH_IN, batch_idx: int) -> Optional[STEP_OUTPUT]:  # type: ignore

        x, y = batch

        pred = torch.tensor(0.0)
        preds: Dict[int, Tensor] = {}
        for i, sample in self.sample_container.items():
            self.sampler.samplable.state = sample
            preds[i] = self.model.predict(x)
            pred = pred + preds[i]

        pred /= len(self.sample_container)
        self.log(f"err/test", self.test_metric(pred, y), prog_bar=True)

        return {"per_sample_predictions": preds, "predictions": pred}

    # def get_sample_logits(self):
    #     return {
    #         i: sum(x.sum() for x in logliks.values())
    #         for i, logliks in self.val_joint_logliks.items()
    #     }


# def draw_n(logits: List[float], n: int) -> List[int]:

#     if n == -1:
#         n = len(logits)

#     out = []
#     logits_t = torch.tensor(logits)
#     for _ in range(n):
#         i = torch.distributions.Categorical(logits=logits_t).sample().item()
#         out.append(i)
#         logits_t[i] = -float("inf")

#     return out
