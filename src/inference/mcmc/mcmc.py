import logging

import torch
from torch.distributions import Gamma
from src.bayesian.modules import BayesianModule
from src.bayesian.priors import NormalPrior

from src.inference.base import InferenceModule
from src.inference.mcmc.samplable import ParameterPosterior, iter_bayesian_modules
from src.inference.mcmc.sample_containers import (
    CompleteSampleContainer,
    FIFOSampleContainer,
    SampleContainer,
)
from src.inference.mcmc.samplers import SGHMC
from src.inference.mcmc.variance_estimators import NextEpochException, NoStepException
from src.models.mlp import MLPClassifier
from src.utils import ParameterView
from src.bayesian.core import to_bayesian_model
log = logging.getLogger(__name__)


class MCMCInference(InferenceModule):
    def __init__(
        self,
        model,
        sampler=None,
        sample_container: SampleContainer = None,
        burn_in=0,
        steps_per_sample=None,
        prior_config=None,
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

        self.val_metrics = torch.nn.ModuleDict(self.model.get_metrics())

    def configure_optimizers(self):
        pass

    def setup(self, stage) -> None:

        self.sampler.setup(self.posterior)
        if not self.sampler.is_batched:
            self.trainer.datamodule.batch_size = None

        self.val_preds = {}

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

            if batch_idx in self.val_preds[i]:
                pred += self.val_preds[i][batch_idx]

            else:
                self.sampler.samplable.state = sample
                pred_ = self.model.predict(x)
                pred += pred_
                self.val_preds[i][batch_idx] = pred_

        pred /= len(self.sample_container)

        for name, metric in self.val_metrics.items():
            self.log(f"{name}/val", metric(pred, y), prog_bar=True)

    def validation_epoch_end(self, outputs) -> None:

        # Delete predictions no longer in use
        delete_keys = set(self.val_preds) - set(self.sample_container.samples)
        for key in delete_keys:
            del self.val_preds[key]
