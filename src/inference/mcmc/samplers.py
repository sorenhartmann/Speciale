from abc import ABC, abstractmethod, abstractproperty
import argparse

from torch.functional import Tensor
from src.utils import HPARAM, HyperparameterMixin, register_component
import typing
from torch.utils.data.dataset import Dataset
from typing import Dict, Generator, Generic, Type, TypeVar, Union, Any, Container
import torch
from torch.distributions import Normal
from torch.utils.data import DataLoader
import inspect

@torch.no_grad()
def clone_parameters(model):
    return {k: x.clone() for k, x in model.named_parameters()}

class Samplable(ABC):

    @abstractproperty
    def state(self) -> torch.Tensor:
        pass

    @abstractmethod
    def prop_log_p(self) -> torch.Tensor:
        pass

    @abstractmethod
    def grad_prop_log_p(self) -> torch.Tensor:
        pass


class Sampler(ABC, HyperparameterMixin):

    is_batched: bool
    tag: str

    @abstractmethod
    def setup(self, samplable: Samplable):
        pass

    @abstractmethod
    def next_sample(self):
        pass


class MetropolisHastings(Sampler):

    step_size: HPARAM[float]

    is_batched = False
    tag = "mh"

    def __init__(self, step_size=0.01):

        self.step_size = step_size

    def setup(self, model):

        self.model = model
        self.state = clone_parameters(model)
        self.log_p = None

    @torch.no_grad()
    def next_sample(self):

        if self.log_p is None:
            self.log_p = self.model.prop_log_p()

        for param in self.model.parameters():
            param.copy_(Normal(param, self.step_size).sample())

        new_log_p = self.model.prop_log_p()
        log_ratio = new_log_p - self.log_p

        if log_ratio > 0 or torch.bernoulli(log_ratio.exp()):
            self.state = clone_parameters(self.model)
            self.log_p = new_log_p
            return self.state

        else:
            self.model.load_state_dict(self.state, strict=False)
            return self.state


class Hamiltonian(Sampler):
    """
    M = I for now...
    """

    step_size: HPARAM[float]
    n_steps: HPARAM[int]

    is_batched = False
    tag = "hmc"

    def __init__(self, step_size=0.01, n_steps=1) -> None:

        self.step_size = step_size
        self.n_steps = n_steps

    def setup(self, samplable: Samplable):

        self.samplable = samplable
        self.momentum = torch.empty_like(self.samplable.state)

        return self

    def U(self):
        return -self.samplable.prop_log_p()

    def grad_U(self):
        grad = self.samplable.grad_prop_log_p()
        return -grad

    def H(self,):
        return self.U() + self.momentum.square().sum() / 2

    def resample_momentum(self):
        self.momentum.normal_()

    def step_momentum(self, half_step=False):

        self.momentum.copy_(
            self.momentum
            - self.step_size * (1.0 if not half_step else 0.5) * self.grad_U()
        )

    def step_parameters(self):
        self.samplable.state = self.samplable.state + self.step_size * self.momentum

    def next_sample(self):

        self.resample_momentum()

        initial_state = self.samplable.state.clone()
        initial_H = self.H()

        self.step_momentum(half_step=True)
        for i in range(self.n_steps):
            self.step_parameters()
            self.step_momentum(half_step=(i == self.n_steps - 1))

        proposed_H = self.H()
        log_acceptance = initial_H - proposed_H

        if log_acceptance >= 0 or log_acceptance.exp() > torch.rand(1):
            # Accepted
            return self.samplable.state.clone()

        else:
            # Rejected
            self.samplable.state = initial_state
            return initial_state

class HamiltonianNoMH(Hamiltonian):

    def next_sample(self):

        self.resample_momentum()
        self.step_momentum(half_step=True)
        for i in range(self.n_steps):
            self.step_parameters()
            self.step_momentum(half_step=(i == self.n_steps - 1))

        return self.samplable.state.clone()


@register_component("sghmc")
class StochasticGradientHamiltonian(Hamiltonian):

    alpha: HPARAM[float]
    beta: HPARAM[float]
    eta: HPARAM[float]

    is_batched = True
    tag = "sghmc"

    def __init__(self, alpha=1e-2, beta=0.0, lr=0.2e-5):

        self.alpha = torch.tensor(alpha)
        self.beta = torch.tensor(beta)
        self.eta = torch.tensor(lr)

        self.err_std = torch.sqrt(2 * (self.alpha - self.beta) * self.eta)

    def setup(self, samplable: Samplable):

        self.samplable = samplable
        self.nu = torch.empty_like(self.samplable.state)
        return self

    def step_parameters(self):

        self.samplable.state = (
            self.samplable.state + self.nu
        )

    def step_nu(self):

        self.nu.copy_(
            self.nu
            - self.eta * self.grad_U()
            - self.alpha * self.nu
            + torch.randn_like(self.nu) * self.err_std
        )
        pass

    def next_sample(self, return_sample=True):

        self.step_nu()
        self.step_parameters()

        if return_sample:
            return self.samplable.state.clone()
