from src.inference import MonteCarloInference
from src.samplers import Hamiltonian, MetropolisHastings
import torch
from torch import nn
from src.modules import BayesianLinearKnownPrecision, BayesianModel, BayesianLinear

class PolynomialToyModel(BayesianModel):

    def __init__(self, coeffs: torch.Tensor):

        super().__init__()

        self.linear = BayesianLinearKnownPrecision(len(coeffs)-1, 1)
        self.linear.setup_prior(1.)

        with torch.no_grad():
            self.linear.bias.copy_(coeffs[0])
            self.linear.weight.copy_(coeffs[1:])

    def forward(self, x: torch.Tensor):

        x = torch.cat([x ** i for i in range(1, self.linear.in_features + 1)], dim=-1)
        return self.linear(x)

    def generate_data(self, n: int = 100, seed: int = 123):
        """
        Generates data point using the polynomial observation model with gaussian noise
        """
        with torch.random.fork_rng():
            with torch.no_grad():
                torch.manual_seed(seed)
                xx = torch.distributions.Uniform(-2.5, 2.5).sample((n, 1))
                yy = self.observation_model(xx).sample()

        return torch.utils.data.TensorDataset(xx, yy)

    def observation_model(self, x):
        mu = self.forward(x)
        return torch.distributions.Normal(mu, 1.)

    def log_likelihood(self, x: torch.FloatTensor, y: torch.FloatTensor):
        """Returns log p(y | x, theta)"""
        return self.observation_model(x).log_prob(y).sum()


class ClassifierNet(BayesianModel):

    def __init__(self, in_features, out_features, hidden_layers=[5]):

        super().__init__()

        layers = []
        in_size = in_features
        for hidden_size in hidden_layers:
            out_size = hidden_size
            layers.append(BayesianLinear(in_size, out_size).setup_prior(2.0, 2.0))
            layers.append(nn.Sigmoid())
            in_size = out_size

        layers.append(BayesianLinear(in_size, out_features).setup_prior(2.0, 2.0))

        self.ffnn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.ffnn(x)

    def log_likelihood(self, x: torch.FloatTensor, y: torch.LongTensor):
        """Returns log p(y | x, theta)"""
        logits = self.forward(x)
        observation_model = torch.distributions.Categorical(logits=logits)
        return observation_model.log_prob(y).sum()


if __name__ == "__main__":

    torch.manual_seed(123)
    true_coeffs = torch.tensor([1.0, 2.0, 0.0, -1.0])

    with torch.no_grad():
        toy_model = PolynomialToyModel(true_coeffs)
        toy_data = toy_model.generate_data(n=1000)   
        for param in toy_model.parameters():
            param.normal_()

    sampler = Hamiltonian(step_size=0.001, n_steps=5)
    inference = MonteCarloInference(
        sampler=sampler, 
        burn_in=2000, 
        n_samples=3000,
        )
    inference.fit(toy_model, toy_data)

    print(inference.sample_df().describe(percentiles=[0.025, 0.975]))