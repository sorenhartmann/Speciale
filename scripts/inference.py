import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

@hydra.main("../conf", "inference_config")
def main(cfg):

    torch.manual_seed(cfg.seed)
    dm = instantiate(cfg.data)
    inference = instantiate(cfg.inference)

    extra_callbacks = cfg.get("extra_callbacks")
    if extra_callbacks is not None:
        OmegaConf.update(
            cfg, 
            "trainer.callbacks", 
            extra_callbacks + cfg.trainer.callbacks
            )

    trainer = instantiate(cfg.trainer)
    trainer.fit(inference, dm)
    return trainer.logged_metrics.get("err/val")
    
if __name__ == "__main__":
    main()