import pytorch_lightning as pl
import torch

from src.modules import BayesianModel
from src.samplers import Sampler
from src.utils import HPARAM, HyperparameterMixin
import pandas as pd


class BayesianRegressor(pl.LightningModule, HyperparameterMixin):

    burn_in: HPARAM[int]
    keep_samples: HPARAM[int]
    use_every: HPARAM[int]

    def __init__(
        self,
        model: BayesianModel,
        sampler: Sampler,
        burn_in=50,
        keep_samples=50,
        use_every=10,
        log_samples=False,
    ):

        super().__init__()

        self.burn_in = burn_in
        self.keep_samples = keep_samples
        self.use_every = use_every

        self.model = model
        self.sampler = sampler

        self.automatic_optimization = False
        self.log_samples = log_samples

        self.save_hyperparameters(self.get_hparams())
        self.save_hyperparameters(self.model.get_hparams())
        self.save_hyperparameters({"sampler": self.sampler.tag})
        self.save_hyperparameters(self.sampler.get_hparams())

        self.samples_ = []

    def configure_optimizers(self):
        return None

    def setup(self, stage) -> None:

        self.sampler.setup(self.model)
        if not self.sampler.is_batched:
            self.trainer.datamodule.batch_size = None

    def training_step(self, batch, batch_idx):

        sample = self.sampler.next_sample(batch)

        if self.global_step < self.burn_in:
            # Burn in sample
            return

        if (self.burn_in + self.global_step) % self.use_every != 0:
            # Thin sample
            return

        if len(self.samples_) == self.keep_samples:
            # Discard oldest sample
            del self.samples_[0]

        if self.log_samples:
            self.log_sample(sample)

        self.samples_.append(sample)

    def validation_step(self, batch, batch_idx):

        if (
            len(self.samples_) == 0
            or (self.burn_in + self.global_step) % self.use_every != 0
        ):
            return

        with torch.no_grad():

            x, y = batch
            pred_samples = []
            for sample in self.samples_:
                self.model.flat_params = sample
                pred_samples.append(self.model.forward(x))

            y_hat = torch.stack(pred_samples).mean(0)

            self.log("loss/val_mse", torch.nn.functional.mse_loss(y_hat, y))

    # def sample_df(self):

    #     tmp = []
    #     for sample in self.samples_:
    #         tmp.append({})
    #         for name, param in sample.items():
    #             tmp[-1].update(
    #                 {
    #                     f"{name}.{i}": value.item()
    #                     for i, value in enumerate(param.flatten())
    #                 }
    #             )

    #     return pd.DataFrame(tmp)

    def log_sample(self, sample):

        for (k, _), (a, b) in zip(
            self.model.param_shapes.items(), self.model.flat_index_pairs
        ):
            for i in range(b - a):
                self.log(f"weights/{k}.{i}", sample[a + i])
