from abc import ABC, abstractmethod
from typing import Generator

import torch
from tqdm import tqdm

from src.models.common import BaysianModule, flatten_tensors


class Sampler(ABC):
    @abstractmethod
    def sample_posterior(
        model: BaysianModule,
        dataset: torch.utils.data.Dataset,
    ) -> Generator[torch.Tensor, None, None]:
        """Method for sampling"""


class MetropolisHastings(Sampler):
    def __init__(self, step_size=0.01):
        self.step_size = step_size

    def sample_posterior(
        self,
        model: BaysianModule,
        dataset: torch.utils.data.Dataset,
    ) -> Generator[torch.Tensor, None, None]:

        try:

            x, y = dataset[:]

            with torch.no_grad():  # No gradients needed

                # log joint probability of current state
                state = model.theta()
                log_p = model.log_prior() + model.log_likelihood(x, y)

                while True:

                    new_state = torch.distributions.Normal(
                        state, self.step_size
                    ).sample()
                    model.update_theta(new_state)
                    new_log_p = model.log_prior() + model.log_likelihood(x, y)
                    log_ratio = new_log_p - log_p

                    if log_ratio > 0 or torch.bernoulli(log_ratio.exp()):

                        state = new_state
                        log_p = new_log_p

                        yield state

        finally:
            pass


def _potential_energy(model, x, y):
    return -model.log_prior() - model.log_likelihood(x, y)


def _potential_energy_grad(model, x, y):
    model.zero_grad()
    _potential_energy(model, x, y).backward()
    return flatten_tensors([x.grad for x in model.parameters()])


def _hamiltonian_function(model, x, y, r):
    return _potential_energy(model, x, y) + torch.dot(r, r) / 2


class Hamiltonian(Sampler):
    """
    M = I for now...
    """

    def __init__(self, step_size=0.01, n_steps=2):

        self.step_size = step_size
        self.n_steps = n_steps

    def sample_posterior(
        self, model: BaysianModule, dataset: torch.utils.data.Dataset
    ) -> Generator[torch.Tensor, None, None]:

        x, y = dataset[:]

        try:

            while True:

                theta = model.theta().detach()
                r = torch.empty_like(theta)
                r.normal_(0, 1)

                with torch.no_grad():
                    H_old = _hamiltonian_function(model, x, y, r)

                theta_old = theta

                r = r - self.step_size / 2 * _potential_energy_grad(model, x, y)

                # Simulate dynamics
                for _ in range(self.n_steps):

                    theta = theta + self.step_size * r
                    model.update_theta(theta)
                    r = r - self.step_size * _potential_energy_grad(model, x, y)

                r = r - self.step_size / 2 * _potential_energy_grad(model, x, y)

                with torch.no_grad():

                    H_hat = _hamiltonian_function(model, x, y, r)
                    diff = H_hat - H_old

                    if diff > 0 or diff.exp() > torch.rand(1):
                        yield theta
                    else:
                        model.update_theta(theta_old)

        finally:
            pass


class StochasticGradientHamiltonian(Sampler):
    """
    M = I for now...
    """

    def __init__(self, step_size=0.01, n_steps=5, batch_size=16):

        self.step_size = step_size
        self.n_steps = n_steps

    def sample_posterior(
        self, model: BaysianModule, dataset: torch.utils.data.Dataset
    ) -> Generator[torch.Tensor, None, None]:

        data_loader = torch.utils.data.DataLoader

        try:

            while True:

                theta = model.theta().detach()
                r = torch.empty_like(theta)
                r.normal_(0, 1)

                with torch.no_grad():
                    H_old = _hamiltonian_function(model, x, y, r)

                theta_old = theta

                r = r - self.step_size / 2 * _potential_energy_grad(model, x, y)

                # Simulate dynamics
                for _ in range(self.n_steps):

                    theta = theta + self.step_size * r
                    model.update_theta(theta)
                    r = r - self.step_size * _potential_energy_grad(model, x, y)

                r = r - self.step_size / 2 * _potential_energy_grad(model, x, y)

                with torch.no_grad():

                    H_hat = _hamiltonian_function(model, x, y, r)
                    diff = H_hat - H_old

                    if diff > 0 or diff.exp() > torch.rand(1):
                        yield theta
                    else:
                        model.update_theta(theta_old)

        finally:
            pass
