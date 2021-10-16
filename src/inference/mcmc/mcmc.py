
import torch
from torch.distributions import Gamma

from src.inference.base import InferenceModule
from src.inference.mcmc.samplable import ParameterPosterior
from src.inference.mcmc.sample_containers import FIFOSampleContainer
from src.inference.mcmc.samplers import SGHMC
from src.inference.mcmc.var_estimators import NoStepException
from src.inference.probabilistic import (KnownPrecisionNormalPrior,
                                         ModuleWithPrior,
                                         to_probabilistic_model_)
from src.models.mlp import MLPClassifier
from src.utils import ParameterView_


class MCMCInference(InferenceModule):

    def __init__(
        self, 
        model, 
        sampler=None, 
        sample_container=None, 
        burn_in=0,
        steps_per_sample=None
        ):

        super().__init__()

        if sampler is None:
            sampler = SGHMC()

        if sample_container is None:
            sample_container = FIFOSampleContainer(max_items=10, keep_every=1)

        self.automatic_optimization = False

        self.model = model
        to_probabilistic_model_(self.model)
        self.view = ParameterView_(self.model)

        self.posterior = ParameterPosterior(self.model)
        self.sampler = sampler

        self.sample_container = sample_container

        self.burn_in = burn_in
        self.steps_per_sample = steps_per_sample

        self.val_metrics = self.model.get_metrics()

        self._skip_val = True



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

    def training_step(self, batch, batch_idx):

        x, y = batch
        sampling_fraction = len(x) / len(self.trainer.train_dataloader.dataset)
        with self.posterior.observe(x, y, sampling_fraction):
            try:
                self.sampler.next_sample(return_sample=False)
            except NoStepException:
                return

        # Only procceed after last batch
        if self.step_until_next_sample > 0:
            self.step_until_next_sample -= 1
            return
        else:
            self.step_until_next_sample = self.steps_per_sample

        self._precision_gibbs_step()

        # Burn in
        if self.burn_in_remaining > 0:
            self.burn_in_remaining =- 1
            return

        # Sample is pruned
        if not self.sample_container.can_use_next():
            self.sample_container.stream_position += 1
            return

        # Save sample
        sample = self.posterior.state.clone().detach()
        self.sample_container.append(sample)
        self._skip_val = False

    def _precision_gibbs_step(self):

        for module in self.model.modules():

            if not isinstance(module, ModuleWithPrior):
                continue

            for name, prior in module.priors.items():
                # FIXME: Hardcoded af

                if not isinstance(prior, KnownPrecisionNormalPrior):
                    continue

                parameter = getattr(module, name)
                alpha = 1.0 + parameter.numel() / 2
                beta = 1.0 + parameter.square().sum() / 2
                new_precision = Gamma(alpha, beta).sample()
                prior.precision.copy_(new_precision)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):

        if self._skip_val:
            return None

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

        self._skip_val = True


if __name__ == "__main__":

    # model = 
    # sampler = StochasticGradientHamiltonian()
    # sample_container = FIFOSampleContainer(max_items=750, keep_every=1)
    inference = MCMCInference(MLPClassifier(hidden_layers=[100, 100]))




    # datamodule = MNISTDataModule(500)



    # Trainer(max_epochs=800).fit(inference, datamodule)



            # yield 

        

        


    