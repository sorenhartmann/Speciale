import hydra
import torch
from hydra.utils import instantiate
from pytorch_lightning import Trainer

from src.data.mnist import MNISTDataModule


@hydra.main("../../conf", "config")
def experiment(cfg):

    torch.manual_seed(123)
    dm = instantiate(cfg.data)
    assert type(dm) is MNISTDataModule

    inference = instantiate(cfg.inference)
    trainer = instantiate(cfg.trainer)
    
    trainer.fit(inference, dm)

if __name__ == "__main__":
    
    experiment()
