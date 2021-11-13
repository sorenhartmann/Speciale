import torch
from pytorch_lightning import LightningDataModule
from torch.distributions import Distribution

from src.data.common import NoBatchMixin
from src.models.base import Model
from src.models.polynomial import PolynomialModel


class SampledData(LightningDataModule, NoBatchMixin):

    def __init__(
        self,
        model : Model,
        covariate_model: Distribution,
        batch_size: int = 8,
        seed=123,
        train_obs=500,
        val_obs=200,
        test_obs=200,
    ):

        super().__init__()

        self.batch_size = batch_size
        self.model = model
        self.covariate_model = covariate_model
        self.seed = seed

        self.num_workers = 0

        self.train_obs = train_obs
        self.val_obs = val_obs
        self.test_obs = test_obs

    def setup(self, stage: str = None):

        with torch.random.fork_rng():

            torch.manual_seed(self.seed)

            self.train_data = self.generate_data(n=self.train_obs)
            self.val_data = self.generate_data(n=self.val_obs)
            self.test_data = self.generate_data(n=self.test_obs)

    def generate_data(self, n):

        xx = self.covariate_model.sample((n,))
        yy = self.model.observation_model(xx).sample()
        return torch.utils.data.TensorDataset(xx, yy)

    def train_dataloader(self):
        return self.get_dataloader(self.train_data)

    def val_dataloader(self):
        return self.get_dataloader(self.val_data)

    def test_dataloader(self):
        return self.get_dataloader(self.test_data)


# if __name__ == "__main__":

#     model = PolynomialModel([1.0, 2.0, 0.0, -1.0])
#     data = SampledData(model, torch.distributions.Uniform(-2.5, 2.5))
#     data.setup()

