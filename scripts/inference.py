import hydra
import torch
from hydra.utils import instantiate

@hydra.main("../conf", "inference_config")
def main(cfg):

    torch.manual_seed(cfg.seed)
    dm = instantiate(cfg.data)
    inference = instantiate(cfg.inference)
    trainer = instantiate(cfg.trainer)
    trainer.fit(inference, dm)
    return trainer.logged_metrics.get("err/val")
    
if __name__ == "__main__":
    main()