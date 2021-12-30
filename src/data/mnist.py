from typing import Any, Dict, Tuple

import torch
from torchvision.datasets import MNIST

from src.data.cifar import ROOT_DIR


class MNISTDataset(MNIST):
    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        super().__init__(root=ROOT_DIR / "data" / "raw", **kwargs)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        img = img.to(torch.float)
        img /= 126.0

        return img, target
