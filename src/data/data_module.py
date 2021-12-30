from typing import Any, Dict, List, Optional, Protocol, Sized, Type, cast

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset, random_split


class DownloadableDatasetProtocol(Protocol):
    def __init__(
        self,
        download: bool = True,
        train: bool = True,
        **data_kwargs: Dict[str, Any],
    ) -> None:
        ...


class DownloadableDataset(DownloadableDatasetProtocol, Dataset):
    ...


class DataModule(LightningDataModule):
    def __init__(
        self,
        dataset_cls: Type[Dataset],
        split_lengths: list[int],
        split_seed: int = 123,
        batch_size: int = 8,  # If -1 return all for each batch
        num_workers: int = 0,
        **data_kwargs: Dict[str, Any],
    ):

        super().__init__()

        self.dataset_cls = dataset_cls
        self.split_lengths = split_lengths
        self.split_seed = split_seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_kwargs = data_kwargs

    def prepare_data(self) -> None:

        # try and download each datasetdownload
        dataset_cls = cast(Type[DownloadableDataset], self.dataset_cls)

        try:
            dataset_cls(download=True, train=True, **self.data_kwargs)
        except Exception:
            pass

        try:
            dataset_cls(download=True, train=False, **self.data_kwargs)
        except Exception:
            pass

    def setup(self, stage: Optional[str] = None) -> None:

        dataset_cls = cast(Type[DownloadableDataset], self.dataset_cls)

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            dataset = dataset_cls(download=False, train=True, **self.data_kwargs)
            self.train_data, self.val_data = self.split_data(dataset)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            test_data = dataset_cls(download=False, train=False, **self.data_kwargs)
            self.test_data = cast(Dataset, test_data)

    def split_data(self, dataset: Dataset) -> List[Subset]:

        return random_split(
            dataset,
            self.split_lengths,
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self) -> DataLoader:

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

    def val_dataloader(self) -> DataLoader:

        if self.batch_size == -1:
            batch_size = len(self.val_data)
        else:
            batch_size = self.batch_size

        return DataLoader(
            self.val_data,
            batch_size=batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:

        if self.batch_size == -1:
            batch_size = len(cast(Sized, self.test_data))
        else:
            batch_size = self.batch_size

        return DataLoader(
            self.test_data,
            batch_size=batch_size,
            num_workers=self.num_workers,
        )
