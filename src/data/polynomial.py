from pytorch_lightning import LightningDataModule
from src.data.common import NoBatchMixin
from src.models.polynomial import PolynomialModel

import torch


class PolynomialDataModule(LightningDataModule, NoBatchMixin):

    coeffs = [1.0, 2.0, 0.0, -1.0]
    seed = 123

    def __init__(self, batch_size: int = 8, train_obs=50, val_obs=20, test_obs=20):

        super().__init__()

        self.batch_size = batch_size
        self.num_workers = 4

        self.train_obs = train_obs
        self.val_obs = val_obs
        self.test_obs = test_obs

    def setup(self, stage: str = None):

        with torch.random.fork_rng():

            torch.manual_seed(self.seed)
        
            toy_model = PolynomialModel(torch.tensor(self.coeffs))

            self.train_data = toy_model.generate_data(n=self.train_obs)
            self.val_data = toy_model.generate_data(n=self.val_obs)
            self.test_data = toy_model.generate_data(n=self.test_obs)

    def train_dataloader(self):
        return self.get_dataloader(self.train_data)

    def val_dataloader(self):
        return self.get_dataloader(self.val_data)

    def test_dataloader(self):
        return self.get_dataloader(self.test_data)
