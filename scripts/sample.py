import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.inference.mcmc.samplable import Samplable
from src.inference.mcmc.samplers import Sampler


@hydra.main("../conf", "sample_config")
def main(cfg: DictConfig) -> None:

    distribution: Samplable = instantiate(cfg.distribution)
    sampler: Sampler = instantiate(cfg.inference.sampler)
    sampler.setup(distribution)
    samples = torch.tensor([sampler.next_sample() for _ in range(cfg.n_samples)])
    torch.save(samples, "samples.pt")


if __name__ == "__main__":
    main()
