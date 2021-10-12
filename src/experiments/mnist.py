import hydra
import torch
from hydra.utils import instantiate

from src.data.mnist import MNISTDataModule


@hydra.main("../../conf", "experiment/mnist/config")
def experiment(cfg):

    torch.manual_seed(123)
    dm = instantiate(cfg.data)

    inference = instantiate(cfg.inference)
    trainer = instantiate(cfg.trainer)
    
    trainer.fit(inference, dm)

    return trainer.callback_metrics["err/val"].item()
    

if __name__ == "__main__":


    experiment()
