import pytest

from src.data.mnist import MNISTDataModule
from src.models.mlp import MLPClassifier


@pytest.fixture
def datamodule():
    dm = MNISTDataModule()
    return dm


@pytest.fixture
def batch(datamodule):
    datamodule.setup()
    return next(iter(datamodule.train_dataloader()))


@pytest.fixture
def classifier():
    return MLPClassifier()
