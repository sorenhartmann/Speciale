from typing import Any, Tuple

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from src.data.cifar import ROOT_DIR


class MNIST(MNIST):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        img = img.to(torch.float)
        img /= 126.

        return img, target


class MNISTDataModule(pl.LightningDataModule):

    data_dir = ROOT_DIR / "data" / "raw"

    def __init__(self, batch_size=32, num_workers=0):

        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dims = (1, 28, 28)

    def prepare_data(self):

        # download
        MNIST(self.data_dir, download=True, train=True)
        MNIST(self.data_dir, download=True, train=False)

    def setup(self, stage: str = None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=ToTensor())
            with torch.random.fork_rng():
                torch.manual_seed(123)
                self.mnist_train, self.mnist_val = random_split(
                    mnist_full, [50000, 10000]
                )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=ToTensor())

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)
