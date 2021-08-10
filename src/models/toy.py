import matplotlib.pyplot as plt
import seaborn as sns
import torch
from src.inference import MonteCarloInference
from src.models.common import BaysianModule
from src.samplers import Hamiltonian, MetropolisHastings
from torch import nn


class PolynomialToyModel(BaysianModule):
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


class NetToyModel(BaysianModule):
    def __init__(self, in_features=1, out_features=1):

        super().__init__()

        self.ffnn = nn.Sequential(
            nn.Linear(in_features, 5),
            nn.Sigmoid(),
            nn.Linear(5, out_features),
        )

        in_features=in_features
        out_features=out_features

    def forward(self, x: torch.Tensor):
        return self.ffnn(x)

    def generate_data(self, n: int = 100, seed: int = 123):
        """
        Generates data point using the polynomial observation model with gaussian noise
        """
        with torch.random.fork_rng():
            with torch.no_grad():
                torch.manual_seed(seed)
                xx = torch.distributions.Uniform(-2.5, 2.5).sample((n, 1))
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
