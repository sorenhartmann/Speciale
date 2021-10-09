from pathlib import Path
from typing import Any, Tuple

import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10

ROOT_DIR = Path(__file__).parents[2]


class CIFARDataModule(pl.LightningDataModule):

    data_dir = ROOT_DIR / "data" / "raw"

    def __init__(self, batch_size=32):

        super().__init__()

        self.batch_size = batch_size
        self.dims = (3, 32, 32)

    def prepare_data(self):

        # download
        CIFAR10(self.data_dir, download=True, train=True)
        CIFAR10(self.data_dir, download=True, train=False)

    def setup(self, stage: str = None):

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            cifar_full = CIFAR10(self.data_dir, train=True, transform=transform)
            with torch.random.fork_rng():
                torch.manual_seed(123)
                self.cifar_train, self.cifar_val = random_split(
                    cifar_full, [40000, 10000]
                )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.cifar_test = CIFAR10(self.data_dir, train=False, transform=transform)

    def train_dataloader(self):
        return DataLoader(
            self.cifar_train, batch_size=self.batch_size, num_workers=0, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size, num_workers=0)
