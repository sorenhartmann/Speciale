import math
import torch
from torch import nn

from src.models.common import BaysianModule
from src.samplers import Sampler

from tqdm import trange

class MonteCarloInference:

    def __init__(self, sampler: Sampler):

        self.sampler = sampler
        self.samples_ = None

    def fit(
        self,
        model: BaysianModule,
        dataset: torch.utils.data.Dataset,
        burn_in=2000,
        n_samples=3000,
    ):

        sample_generator = self.sampler.sample_posterior(model, dataset)

        for i in trange(burn_in, desc="Burn in:"):
            next(sample_generator)

        n_params = sum(math.prod(x.shape) for x in model.parameters())
        self.samples_ = torch.empty((n_samples, n_params))

        for i in trange(n_samples, desc="Sampling...:"):
            self.samples_[i, :] = next(sample_generator)

        sample_generator.close()
