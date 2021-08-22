import torch
from src.modules import BayesianLinearKnownPrecision, BayesianModel


class PolynomialModel(BayesianModel):

    def __init__(self, coeffs: torch.Tensor):

        super().__init__()

        self.linear = BayesianLinearKnownPrecision(len(coeffs) - 1, 1)
        self.linear.setup_prior(1.0)

        with torch.no_grad():
            self.linear.bias.copy_(coeffs[0])
            self.linear.weight.copy_(coeffs[1:])

    def forward(self, x: torch.Tensor):

        x = torch.cat([x ** i for i in range(1, self.linear.in_features + 1)], dim=-1)
        return self.linear(x)

    def generate_data(self, n: int = 100):
        """
        Generates data point using the polynomial observation model with gaussian noise
        """
        with torch.no_grad():
            xx = torch.distributions.Uniform(-2.5, 2.5).sample((n, 1))
            yy = self.observation_model(xx).sample()

        return torch.utils.data.TensorDataset(xx, yy)

    def observation_model(self, x):
        mu = self.forward(x)
        return torch.distributions.Normal(mu, 1.0)

    def log_likelihood(self, x: torch.FloatTensor, y: torch.FloatTensor):
        """Returns log p(y | x, theta)"""
        return self.observation_model(x).log_prob(y)
