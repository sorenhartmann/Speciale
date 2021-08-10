
import math
from src.inference import MonteCarloInference
import torch
from torch import nn
from src.models.common import BaysianModule
from src.samplers import Hamiltonian, MetropolisHastings
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import trange

class ToyModel(BaysianModule):

    def __init__(self, coeffs: torch.Tensor, noise_scale=1.0):

        super().__init__()

        self.noise_scale = noise_scale
        self.register_parameter("coeffs", nn.Parameter(coeffs))

    def forward(self, x: torch.Tensor):
        return sum(c * x ** i for i, c in enumerate(self.coeffs))

    def generate_data(self, n: int = 100, seed: int = 123):
        """
        Generates data point using the polynomial observation model with gaussian noise
        """
        with torch.random.fork_rng():
            with torch.no_grad():
                torch.manual_seed(seed)
                xx = torch.distributions.Uniform(-2.5, 2.5).sample((n,))
                yy = self.forward(xx)
                yy = yy + torch.randn_like(yy) * self.noise_scale

        return torch.utils.data.TensorDataset(xx, yy)

    # TODO: Fix noget med navngivningen?
    def log_prior(self):
        """Returns log p(theta)"""
        prior = torch.distributions.Normal(0, 1)
        return sum(prior.log_prob(x).sum() for x in self.parameters())

    def log_likelihood(self, x: torch.FloatTensor, y: torch.FloatTensor):
        """Returns log p(y | x, theta)"""

        mu = self.forward(x)
        observation_model = torch.distributions.Normal(mu, self.noise_scale)
        return observation_model.log_prob(y).sum()

if __name__ == "__main__":

    true_coeffs = torch.tensor([1.0, 2.0, 0.0, -1.0])

    torch.manual_seed(124)

    with torch.no_grad():

        toy_model = ToyModel(true_coeffs)
        toy_data = toy_model.generate_data(n=100)    
        toy_model.coeffs.normal_()

    sampler = Hamiltonian(step_size=0.01, n_steps=5)
    inference = MonteCarloInference(sampler=sampler)
    inference.fit(toy_model, toy_data, burn_in=100, n_samples=500)