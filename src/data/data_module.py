from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import random_split, DataLoader, Dataset
from typing import Type


class DataModule(LightningDataModule):
    def __init__(
        self,
        dataset_cls: Type[Dataset],
        split_lengths: list[int],
        split_seed: int = 123,
        batch_size: int = 8,  # If -1 return all for each batch
        num_workers: int = 0,
        **data_kwargs
    ):

        super().__init__()

        self.dataset_cls = dataset_cls
        self.split_lengths = split_lengths
        self.split_seed = split_seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_kwargs = data_kwargs

    def prepare_data(self):

        # download
        self.dataset_cls(download=True, train=True, **self.data_kwargs)
        try:
            self.dataset_cls(download=True, train=False, **self.data_kwargs)
        except Exception:
            pass

    def setup(self, stage: str = None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            dataset = self.dataset_cls(**self.data_kwargs, train=True)
            self.train_data, self.val_data = self.split_data(dataset)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_data = self.dataset_cls(train=False, **self.data_kwargs)


    def split_data(self, dataset):

        return random_split(
            dataset,
            self.split_lengths,
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self):

        if self.batch_size == -1:
            batch_size = len(self.train_data)
        else:
            batch_size = self.batch_size

        return DataLoader(
            self.train_data,
            batch_size=batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):

        if self.batch_size == -1:
            batch_size = len(self.val_data)
        else:
            batch_size = self.batch_size

        return DataLoader(
            self.val_data,
            batch_size=batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):

        if self.batch_size == -1:
            batch_size = len(self.test_data)
        else:
            batch_size = self.batch_size

        return DataLoader(
            self.test_data,
            batch_size=batch_size,
            num_workers=self.num_workers,
        )
        