import hydra
from hydra.utils import instantiate
import torch

@hydra.main("../conf", "sample_config")
def main(cfg):

    distribution = instantiate(cfg.distribution)
    sampler = instantiate(cfg.inference.sampler)
    sampler.setup(distribution)
    samples = torch.tensor([sampler.next_sample() for i in range(cfg.n_samples)])
    torch.save(samples, "samples.pt")

if __name__ == "__main__":
    main()
