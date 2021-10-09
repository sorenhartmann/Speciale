from contextlib import contextmanager

import torch
import torchmetrics.functional
from torch.distributions import Gamma

from src.inference.base import InferenceModule
from src.inference.mcmc.sample_containers import FIFOSampleContainer
from src.inference.mcmc.samplers import (Samplable, StochasticGradientHamiltonian)
from src.inference.probabilistic import (KnownPrecisionNormalPrior,
                                         ModuleWithPrior, to_probabilistic_model_)
from src.models.mlp import MLPClassifier
from src.utils import ParameterView_


class ParameterPosterior(Samplable):
    """Posterior of model parameters given observations"""

    def __init__(self, model):

        self.model = model
        self.view = ParameterView_(model)

        self._x = None
        self._y = None
        self._sampling_fraction = 1.0

    def prop_log_p(self) -> torch.Tensor:
        return (
            self.model.log_prior()
            + self.model.log_likelihood(x=self._x, y=self._y).sum()
            / self._sampling_fraction
        )

    def grad_prop_log_p(self):
        self.model.zero_grad()
        t = self.prop_log_p()
        t.backward()
        return self.view.flat_grad

    def set_observation(self, x=None, y=None, sampling_fraction: float = 1.0):

        self._x = x
        self._y = y
        self._sampling_fraction = sampling_fraction

    @contextmanager
    def observe(self, x=None, y=None, sampling_fraction: float = 1.0):

        x_prev = self._x
        y_prev = self._y
        sampling_fraction_prev = self._sampling_fraction

        try:
            self.set_observation(x, y, sampling_fraction)
            yield
        finally:
            self.set_observation(x_prev, y_prev, sampling_fraction_prev)

    @property
    def state(self):
        return self.view[:]

    @state.setter
    def state(self, value):
        self.model.load_state_dict(
            {
                k: value[a:b].view(shape)
                for (k, shape), (a, b) in zip(
                    self.view.param_shapes.items(), self.view.flat_index_pairs
                )
            },
            strict=False,
        )

class MCMCInference(InferenceModule):

    def __init__(self, model, sampler=None, sample_container=None, burn_in=0):

        super().__init__()

        if sampler is None:
            sampler = StochasticGradientHamiltonian()

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

        self.val_metrics = self.model.get_metrics()

        self._skip_val = True

    def configure_optimizers(self):
        pass

    def setup(self, stage) -> None:

        self.sampler.setup(self.posterior)
        if not self.sampler.is_batched:
            self.trainer.datamodule.batch_size = None

        self.val_preds = {}

    def training_step(self, batch, batch_idx):

        x, y = batch
        sampling_fraction = len(x) / len(self.trainer.train_dataloader.dataset)
        with self.posterior.observe(x, y, sampling_fraction):
            self.sampler.next_sample(return_sample=False)

        # Only procced after last batch
        if batch_idx < len(self.trainer.train_dataloader) - 1:
            return

        self._precision_gibbs_step()

        # Burn in
        if self.current_epoch < self.burn_in:
            return

        # Sample is pruned
        if not self.sample_container.can_use_next():
            self.sample_container.stream_position += 1
            return

        # Save sample
        sample = self.posterior.state.clone()
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

        probs_sum = torch.zeros((len(x), 10))

        for i, sample in self.sample_container.items():

            if i not in self.val_preds:
                self.val_preds[i] = {}

            if batch_idx in self.val_preds[i]:
                probs = self.val_preds[i][batch_idx]
            else:
                self.posterior.state = sample
                probs = self.model.forward(x).softmax(-1)
                self.val_preds[i][batch_idx] = probs

            probs_sum += probs

        preds = (probs_sum / len(self.sample_container)).argmax(-1)
        self.log(
            "val_err", 1 - torchmetrics.functional.accuracy(preds, y), prog_bar=True
        )

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

        

        


    