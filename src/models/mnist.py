import torch
from torch import nn

from src.data.mnist import MNISTDataModule
from src.samplers import Hamiltonian, StochasticGradientHamiltonian
from src.modules import BayesianLinear, BayesianLinearKnownPrecision, BayesianMixin, BayesianModel
from tqdm import trange
from torch.distributions import Gamma
import math
import pytorch_lightning as pl



class MNISTModel(BayesianModel):

    def __init__(
        self, in_features=784, out_features=10, hidden_layers=[100], alpha=1.0, beta=1.0
    ):
        super().__init__()

        layers = []
        in_size = in_features
        for hidden_size in hidden_layers:
            out_size = hidden_size
            layers.append(
                BayesianLinearKnownPrecision(in_size, out_size).setup_prior(1.)
            )
            layers.append(nn.Sigmoid())
            in_size = out_size

        layers.append(
            BayesianLinearKnownPrecision(in_size, out_features).setup_prior(1.)
        )

        self.ffnn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        x = x.flatten(-2, -1)
        return self.ffnn(x)

    def log_likelihood(self, x: torch.FloatTensor, y: torch.LongTensor):
        """Returns log p(y | x, theta)"""
        logits = self.forward(x)
        observation_model = torch.distributions.Categorical(logits=logits)
        return observation_model.log_prob(y).sum()

def main():


    torch.manual_seed(123)

    dm = MNISTDataModule(500)
    dm.setup()

    alpha, beta = 1., 1.

    train_data, test_data = dm.mnist_train, dm.mnist_test

    model = MNISTModel()
    sampler = StochasticGradientHamiltonian(alpha=0.01, beta=0, eta=2e-6, n_steps=3)
    sampler.setup_sampler(model, train_data)

    for i in trange(800):

        samples = []
        for _ in range(100):
            samples.append(sampler.next_sample())
        
        
        # Update gamma
        with torch.no_grad():

            new_precisions = {}

            for name, param in model.named_parameters():
                param_samples = torch.stack([x[name] for x in samples])
                
                # Iterateively or with prior?
                precision_posterior = Gamma( 
                    alpha + math.prod(param_samples.size()),
                    beta + (param_samples ** 2).sum() / 2 
                )

                new_precisions[name] = precision_posterior.sample()

            print(new_precisions)

            model.ffnn[0].weight_precision.copy_(new_precisions["ffnn.0.weight"])
            model.ffnn[0].bias_precision.copy_(new_precisions["ffnn.0.bias"])
            model.ffnn[2].weight_precision.copy_(new_precisions["ffnn.2.weight"])
            model.ffnn[2].bias_precision.copy_(new_precisions["ffnn.2.bias"])

if __name__ == "__main__":

    main()  


#