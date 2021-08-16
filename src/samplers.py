from abc import ABC, abstractmethod
from torch.nn import parameter
from torch.utils.data.dataset import Dataset
from src.modules import BayesianModel
from typing import Generator, Union

import torch

from torch.distributions import Normal


@torch.no_grad()
def clone_parameters(model):
    return {k: x.clone() for k, x in model.named_parameters()}


class Sampler(ABC):
    def sample_generator(
        self, model: BayesianModel, dataset: Dataset
    ) -> Generator[torch.Tensor, None, None]:

        self.setup_sampler(model, dataset)
        while True:
            yield self.next_sample()

    @abstractmethod
    def setup_sampler(self, model: BayesianModel, dataset: Dataset):
        pass

    @abstractmethod
    def next_sample(self):
        pass


class MetropolisHastings(Sampler):
    def __init__(self, step_size=0.01):

        self.step_size = step_size

    def setup_sampler(self, model: BayesianModel, dataset: Dataset):

        x, y = dataset[:]
        self.x = x
        self.y = y
        self.model = model
        self.state = clone_parameters(model)
        with torch.no_grad():
            self.log_p = self.model.log_joint(self.x, self.y)

    @torch.no_grad()
    def next_sample(self):

        for param in self.model.parameters():
            param.copy_(Normal(param, self.step_size).sample())

        new_log_p = self.model.log_joint(self.x, self.y)
        log_ratio = new_log_p - self.log_p

        if log_ratio > 0 or torch.bernoulli(log_ratio.exp()):
            self.state = clone_parameters(self.model)
            return self.state

        else:
            self.model.load_state_dict(self.state, strict=False)
            return None


class Hamiltonian(Sampler):
    """
    M = I for now...
    """

    def __init__(self, step_size=0.01, n_steps=1):

        self.step_size = step_size
        self.n_steps = n_steps

    def setup_sampler(self, model: BayesianModel, dataset: Dataset):

        x, y = dataset[:]
        self.x = x
        self.y = y
        self.model = model
        self.state = clone_parameters(model)
        self.momentum = {k : torch.empty_like(x) for k, x in self.state.items()}
        
    def U(self):
        return -self.model.log_joint(self.x, self.y)

    @torch.no_grad()
    def H(self):
        result = 0
        for r in self.momentum.values():
            result += torch.dot(r.flatten(), r.flatten()) / 2
        result += self.U()
        return result

    def resample_momentum(self):
        for r in self.momentum.values():
            r.normal_()

    def step_momentum(self, step_size):
        self.model.zero_grad()
        self.U().backward()
        parameters = dict(self.model.named_parameters())
        for name, r in self.momentum.items():
            r.copy_(r - step_size * parameters[name].grad)

    @torch.no_grad()
    def step_parameters(self, step_size):
        for name, param in self.model.named_parameters():
            param.copy_(param + step_size * self.momentum[name])

    def next_sample(self):

        self.resample_momentum()
        H_current = self.H()

        self.step_momentum(self.step_size / 2)
        
        for _ in range(self.n_steps):

            self.step_parameters(self.step_size)
            self.step_momentum(self.step_size)

        self.step_momentum(self.step_size / 2)

        H_proposed = self.H()
        log_rho = H_proposed - H_current

        if log_rho > 0 or torch.rand(1) < log_rho.exp():
            self.state = clone_parameters(self.model)
            return self.state
        else:
            self.model.load_state_dict(self.state, strict=False)
            return None




# class StochasticGradientHamiltonian(Sampler):
#     """
#     M = I for now...
#     """

#     def __init__(self, step_size=0.01, n_steps=5, batch_size=16):

#         self.step_size = step_size
#         self.n_steps = n_steps

#     def sample_posterior(
#         self, model: BayesianModel, dataset: torch.utils.data.Dataset
#     ) -> Generator[torch.Tensor, None, None]:

#         data_loader = torch.utils.data.DataLoader

#         try:

#             while True:

#                 theta = model.theta().detach()
#                 r = torch.empty_like(theta)
#                 r.normal_(0, 1)

#                 with torch.no_grad():
#                     H_old = _hamiltonian_function(model, x, y, r)

#                 theta_old = theta

#                 r = r - self.step_size / 2 * _potential_energy_grad(model, x, y)

#                 # Simulate dynamics
#                 for _ in range(self.n_steps):

#                     theta = theta + self.step_size * r
#                     model.step_theta(theta)
#                     r = r - self.step_size * _potential_energy_grad(model, x, y)

#                 r = r - self.step_size / 2 * _potential_energy_grad(model, x, y)

#                 with torch.no_grad():

#                     H_hat = _hamiltonian_function(model, x, y, r)
#                     diff = H_hat - H_old

#                     if diff > 0 or diff.exp() > torch.rand(1):
#                         yield theta
#                     else:
#                         model.step_theta(theta_old)

#         finally:
#             pass
