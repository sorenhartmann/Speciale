import torch

from src.data.mnist import MNISTDataModule
from src.inference import MonteCarloInference
from src.models.simple import Net
from src.samplers import Hamiltonian

class MNISTModel(Net):

    def forward(self, x):
        x = x.flatten(-2, -1)
        return self.ffnn(x)
    
    def log_likelihood(self, x: torch.FloatTensor, y: torch.LongTensor):
        """Returns log p(y | x, theta)"""
        logits = self.forward(x)
        observation_model = torch.distributions.Categorical(logits=logits)
        return observation_model.log_prob(y).sum()

if __name__ == "__main__":

    dm = MNISTDataModule(32)
    dm.setup()

    train_data, test_data = dm.mnist_train, dm.mnist_test

    model = MNISTModel(784, 10, [100, 100])

    sampler = Hamiltonian(step_size=0.1, n_steps=3)
    inference = MonteCarloInference(sampler=sampler)
    inference.fit(model, train_data, burn_in=20, n_samples=0)