import pytorch_lightning as pl
import torch
from functools import cached_property
from src.samplers import Sampler, Samplable
from src.utils import HPARAM, HyperparameterMixin, pairwise
from contextlib import contextmanager
from src.modules import ProbabilisticModel
import torchmetrics.functional
import torchmetrics.functional as FM
from itertools import accumulate
import math

from tensorboard.backend.event_processing import event_accumulator

class FIFOSampleContainer:
    """Retain as set of samples given an stream of samples of unkown length"""

    def __init__(self, max_items=20, keep_every=20):

        self.max_items = max_items
        self.keep_every = keep_every

        self.samples = {}
        self.stream_position = 0

    def append(self, value):

        if not self.can_use_next():
            self.stream_position += 1
            return

        if len(self.samples) == self.max_items:
            del self.samples[min(self.samples)]

        self.samples[self.stream_position] = value
        self.stream_position += 1

    def can_use_next(self) -> bool:
        return self.stream_position % self.keep_every == 0

    def items(self):
        return self.samples.items()

    def values(self):
        return self.samples.values()

    def __iter__(self):
        return iter(self.samples)

    def __len__(self):
        return len(self.samples)

class ParameterPosterior(Samplable):

    """Posterior of model parameters given observations"""

    def __init__(self, model):

        self.model = model

        self._x = None
        self._y = None
        self._sampling_fraction = 1.0

    def prop_log_p(self) -> torch.Tensor:
        return (
            self.model.log_prior()
            + self.model.log_likelihood(
                x=self._x, y=self._y, sampling_fraction=self._sampling_fraction
            ).sum() 
        )

    def grad_prop_log_p(self):
        self.model.zero_grad()
        self.prop_log_p().backward()
        return self.state_grad

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

    @cached_property
    def param_shapes(self):
        return {k: x.shape for k, x in self.model.named_parameters()}

    @cached_property
    def flat_index_pairs(self):
        indices = accumulate(
            self.param_shapes.values(), lambda x, y: x + math.prod(y), initial=0
        )
        return list(pairwise(indices))

    @property
    def state(self) -> torch.Tensor:
        return torch.cat([x.detach().flatten() for x in self.model.parameters()])

    @state.setter
    def state(self, value):

        self.model.load_state_dict(
            {
                k: value[a:b].view(shape)
                for (k, shape), (a, b) in zip(
                    self.param_shapes.items(), self.flat_index_pairs
                )
            },
            strict=False,
        )

    @property
    def state_grad(self) -> torch.Tensor:
        return torch.cat([x.grad.flatten() for x in self.model.parameters()])

    def flatten(self, tensor_iter):
        return torch.cat([x.flatten() for x in tensor_iter])

    def unflatten(self, value):
        return {
            k: value[a:b].view(shape)
            for (k, shape), (a, b) in zip(
                self.param_shapes.items(), self.flat_index_pairs
            )
        }
        

class VariationalInference(pl.LightningModule, HyperparameterMixin):
    pass
    


class BayesianClassifier(pl.LightningModule, HyperparameterMixin):


    burn_in: HPARAM[int]
    keep_samples: HPARAM[int]

    def __init__(
        self,
        model: ProbabilisticModel,
        sampler: Sampler,
        burn_in=50,  ## Epochs
        keep_samples=800,
    ):

        super().__init__()

        self.burn_in = burn_in
        self.keep_samples = keep_samples

        self.model = model
        self.posterior = ParameterPosterior(self.model)
        self.sampler = sampler

        self.automatic_optimization = False

        self.save_hyperparameters(self.get_hparams())
        self.save_hyperparameters(self.model.get_hparams())
        self.save_hyperparameters({"sampler": self.sampler.tag})
        self.save_hyperparameters(self.sampler.get_hparams())

        self.sample_container = FIFOSampleContainer(keep_samples, keep_every=1)
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
        
        self.model.precision_gibbs_step()

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


    
class MAPClassifier(pl.LightningModule):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def training_step(self, batch, batch_idx):
        x, y = batch
        sampling_fraction = len(x) / len(self.trainer.train_dataloader.dataset)
        loss = -self.model.log_likelihood(x, y).sum() / sampling_fraction - self.model.log_prior()
        # TODO: scale likelihood?
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x).argmax(-1)
        self.log("val_err", 1 - FM.accuracy(y_hat, y), prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=0.2e-5, momentum=0.99)


# class BayesianRegressor(pl.LightningModule, HyperparameterMixin):

#     burn_in: HPARAM[int]
#     keep_samples: HPARAM[int]
#     use_every: HPARAM[int]

#     def __init__(
#         self,
#         model: BayesianModel,
#         sampler: Sampler,
#         burn_in=100,
#         keep_samples=50,
#         use_every=50,
#     ):

#         super().__init__()

#         self.burn_in = burn_in
#         self.keep_samples = keep_samples
#         self.use_every = use_every

#         self.model = model
#         self.sampler = sampler

#         self.automatic_optimization = False

#         self.save_hyperparameters(self.get_hparams())
#         self.save_hyperparameters(self.model.get_hparams())
#         self.save_hyperparameters({"sampler": self.sampler.tag})
#         self.save_hyperparameters(self.sampler.get_hparams())

#         self.samples_ = []

#     def configure_optimizers(self):
#         return None

#     def setup(self, stage) -> None:

#         self.sampler.setup(self.model)
#         if not self.sampler.is_batched:
#             self.trainer.datamodule.batch_size = None

#     def training_step(self, batch, batch_idx):

#         x, y = batch
#         sample = self.sampler.next_sample(x, y)

#         if self.global_step < self.burn_in:
#             # Burn in sample
#             return 

#         if (self.burn_in + self.global_step) % self.use_every != 0:
#             # Thin sample
#             return None

#         if len(self.samples_) == self.keep_samples:
#             # Discard oldest sample
#             del self.samples_[0]

#         self.samples_.append(sample)

#     def validation_step(self, batch, batch_idx):

#         if (
#             len(self.samples_) == 0
#             or (self.burn_in + self.global_step) % self.use_every != 0
#         ):
#             return

#         with torch.no_grad():

#             x, y = batch
#             pred_samples = []
#             for sample in self.samples_:
#                 self.model.flat_params = sample
#                 pred_samples.append(self.model.forward(x))

#             y_hat = torch.stack(pred_samples).mean(0)

#             self.log("loss/val_mse", torch.nn.functional.mse_loss(y_hat, y))

#     # def sample_df(self):

#     #     tmp = []
#     #     for sample in self.samples_:
#     #         tmp.append({})
#     #         for name, param in sample.items():
#     #             tmp[-1].update(
#     #                 {
#     #                     f"{name}.{i}": value.item()
#     #                     for i, value in enumerate(param.flatten())
#     #                 }
#     #             )

#     #     return pd.DataFrame(tmp)


