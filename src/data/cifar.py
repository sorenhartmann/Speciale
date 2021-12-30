from pathlib import Path
from typing import Any, Dict

from torchvision.datasets import CIFAR10

ROOT_DIR = Path(__file__).parents[2]


class CIFAR10Dataset(CIFAR10):
    def __init__(self, **kwargs: Dict[str, Any]):
        super().__init__(root=ROOT_DIR / "data" / "raw", **kwargs)
