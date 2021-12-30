from typing import Optional

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning import Trainer

from src.data.data_module import DataModule
from src.inference.base import InferenceModule


@hydra.main("../conf", "inference_config")
def main(cfg: DictConfig) -> Optional[float]:

    torch.manual_seed(cfg.seed)
    dm: DataModule = instantiate(cfg.data)
    inference: InferenceModule = instantiate(cfg.inference)

    extra_callbacks: Optional[ListConfig] = cfg.get("extra_callbacks")
    if extra_callbacks is not None:
        OmegaConf.update(
            cfg, "trainer.callbacks", extra_callbacks + cfg.trainer.callbacks
        )

    trainer: Trainer = instantiate(cfg.trainer)
    trainer.fit(inference, dm)

    if trainer.interrupted:
        raise KeyboardInterrupt

    if cfg.test:
        trainer.test(inference, dm, ckpt_path=cfg.test_ckpt_path)

    return trainer.logged_metrics.get("err/val")


if __name__ == "__main__":
    main()
