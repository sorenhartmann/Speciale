import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Literal, Union


class AllSampler(torch.utils.data.Sampler):
    
    def __init__(self):
        pass

    def __iter__(self):
        yield ...

    def __len__(self):
        return 1

class NoBatchMixin:

    batch_size: Union[int, Literal[None]]
    num_workers: int

    def get_dataloader(self, dataset):
        
        # TODO: check num_worker on "real" dataset
        if self.batch_size is not None:
            return DataLoader(
                dataset, batch_size=self.batch_size, num_workers=self.num_workers
            )
        else:
            return DataLoader(
                dataset,
                sampler=AllSampler(),
                batch_size=None,
                num_workers=0,
            ) 
