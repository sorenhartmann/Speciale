import logging
from re import S
from typing import Dict, List, Optional
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
import torch
from torch.distributions import Gamma
from torch import Tensor
from src.bayesian.priors import NormalPrior

from src.inference.base import InferenceModule
from src.inference.mcmc.samplable import ParameterPosterior
from src.inference.mcmc.sample_containers import (
    CompleteSampleContainer,
    SampleContainer,
)
from src.inference.mcmc.samplers import SGHMC
from src.inference.mcmc.variance_estimators import NextEpochException, NoStepException
from src.bayesian.core import iter_bayesian_modules, to_bayesian_model
from src.models.base import ErrorRate

log = logging.getLogger(__name__)


class MCMCInference(InferenceModule):
    def __init__(
        self,
        model,
        sampler=None,
        sample_container: SampleContainer = None,
        burn_in=0,
        steps_per_sample=None,
        use_gibbs_step=True,
        prior_config=None,
        filter_samples_before_test: float = 1.0,
    ):

        super().__init__()

        if sampler is None:
            sampler = SGHMC()

        if sample_container is None:
            sample_container = CompleteSampleContainer()

        self.automatic_optimization = False

        self.model = to_bayesian_model(model, prior_config)
        self.posterior = ParameterPosterior(self.model)
        self.sampler = sampler

        self.sample_container = sample_container

        self.burn_in = burn_in
        self.steps_per_sample = steps_per_sample

        self.use_gibbs_step = use_gibbs_step

        self.val_metrics = torch.nn.ModuleDict(self.model.get_metrics())
        self._precision_gibbs_step()

        self.filter_samples_before_test = filter_samples_before_test

    def configure_optimizers(self):
        pass

    def setup(self, stage) -> None:

        self.sampler.setup(self.posterior)


    def on_fit_start(self) -> None:
        self.val_preds = {}
        self.val_joint_logliks = {}
        self.val_avg_likelihood = {}

    def on_train_start(self) -> None:

        if self.steps_per_sample is None:
            self.steps_per_sample = len(self.trainer.train_dataloader)

        self.step_until_next_sample = self.steps_per_sample
        self.burn_in_remaining = self.burn_in

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        if getattr(self, "_finish_train_epoch", False):
            self._finish_train_epoch = False
            return -1

    def on_train_epoch_start(self) -> None:
        self.sampler.on_train_epoch_start(self)

    def training_step(self, batch, batch_idx):

        x, y = batch
        sampling_fraction = len(x) / len(self.trainer.train_dataloader.dataset)

        try:
            with self.posterior.observe(x, y, sampling_fraction):
                self.sampler.next_sample(return_sample=False)
        except NoStepException:
            return
        except NextEpochException:
            self._finish_train_epoch = True
            return

        # Only procceed after last batch
        self.step_until_next_sample -= 1
        is_last_batch = self.step_until_next_sample == 0
        if not is_last_batch:
            return
        else:
            self.step_until_next_sample = self.steps_per_sample

        if self.use_gibbs_step:
            self._precision_gibbs_step()

        # Burn in
        burn_in = self.burn_in_remaining > 0
        if burn_in:
            self.burn_in_remaining -= 1
            return

        # Register sample with sample container. Pruning is handled by container
        def get_sample():
            return self.posterior.state.clone().detach()

        self.sample_container.register_sample(get_sample)

    def _precision_gibbs_step(self):

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
    def validation_step(self, batch, batch_idx):

        if len(self.sample_container) == 0:
            return

        x, y = batch

        pred = 0

        for i, sample in self.sample_container.items():

            if i not in self.val_preds:
                self.val_preds[i] = {}
                self.val_joint_logliks[i] = {}
                self.val_avg_likelihood[i] = {}

            if batch_idx in self.val_preds[i]:
                pred += self.val_preds[i][batch_idx]

            else:
                old_state = self.sampler.samplable.state
                self.sampler.samplable.state = sample
                output = self.model(x)
                pred_ = self.model.predict_gvn_output(output)
                obs_model = self.model.observation_model_gvn_output(output)
                pred += pred_
                self.val_preds[i][batch_idx] = pred_
                log_prob : Tensor =  obs_model.log_prob(y)
                self.val_joint_logliks[i][batch_idx] = log_prob.sum()
                self.val_avg_likelihood[i][batch_idx] = log_prob.exp().mean()
                self.sampler.samplable.state = old_state

        pred /= len(self.sample_container)

        for name, metric in self.val_metrics.items():
            self.log(f"{name}/val", metric(pred, y), prog_bar=True)

    def validation_epoch_end(self, outputs) -> None:

        # Delete predictions no longer in use
        delete_keys = set(self.val_preds) - set(self.sample_container.samples)
        for key in delete_keys:
            del self.val_preds[key]
            del self.val_joint_logliks[key]
            del self.val_avg_likelihood[key]

    def on_test_epoch_start(self) -> None:

        self.test_metric = ErrorRate().to(device=self.device)
        if self.filter_samples_before_test != 1:
            sample_logits = self.get_sample_logits()
            
    @torch.no_grad()
    def test_step(self, batch, batch_idx):

        x, y = batch

        pred = 0
        preds: Dict[int, Tensor] = {}
        for i, sample in self.sample_container.items():
            self.sampler.samplable.state = sample
            preds[i] = self.model.predict(x)
            pred += preds[i]

        pred /= len(self.sample_container)
        self.log(f"err/test", self.test_metric(pred, y), prog_bar=True)

        return {"batch_idx": batch_idx, "predictions": preds, "target": y}


    def get_sample_logits(self):
        return {
            i: sum(x.sum() for x in logliks.values())
            for i, logliks in self.val_joint_logliks.items()
        }

def draw_n(logits: List[float], n: int) -> List[int]:
    

    if n == -1:
        n = len(logits)

    out = []
    logits_t = torch.tensor(logits)
    for _ in range(n):
        i = torch.distributions.Categorical(logits=logits_t).sample().item()
        out.append(i)
        logits_t[i] = -float("inf")
        
    return out
