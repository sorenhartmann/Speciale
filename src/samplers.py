from abc import ABC, abstractmethod, abstractproperty
import argparse

from torch.functional import Tensor
from src.utils import HPARAM, HyperparameterMixin
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
    def next_sample(self, *args):

        if self.log_p is None:
            self.log_p = self.model.prop_log_p(*args)

        for param in self.model.parameters():
            param.copy_(Normal(param, self.step_size).sample())

        new_log_p = self.model.prop_log_p(*args)
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

    def U(self, *args):
        return -self.samplable.prop_log_p(*args)

    def grad_U(self, *args):
        return -self.samplable.grad_prop_log_p(*args)

    def H(self, *args):
        return self.U(*args) + self.momentum.square().sum() / 2

    def resample_momentum(self):
        self.momentum.normal_()

    def step_momentum(self, *args, half_step=False):

        self.momentum.copy_(
            self.momentum
            - self.step_size * (1.0 if not half_step else 0.5) * self.grad_U(*args)
        )

    def step_parameters(self):
        self.samplable.state = self.samplable.state + self.step_size * self.momentum

    def next_sample(self, *args):

        self.resample_momentum()

        initial_state = self.samplable.state.clone()
        initial_H = self.H(*args)

        self.step_momentum(*args, half_step=True)
        for i in range(self.n_steps):
            self.step_parameters()
            self.step_momentum(*args, half_step=(i == self.n_steps - 1))

        proposed_H = self.H(*args)
        log_acceptance = initial_H - proposed_H

        if log_acceptance >= 0 or log_acceptance.exp() > torch.rand(1):
            # Accepted
            return self.samplable.state.clone()

        else:
            # Rejected
            self.samplable.state = initial_state
            return initial_state


class StochasticGradientHamiltonian(Sampler):
    """
    M = I for now... Using diagonal variance estimate
    """

    alpha: HPARAM[float]
    beta: HPARAM[float]
    eta: HPARAM[float]
    n_steps: HPARAM[int]

    is_batched = True
    tag = "sghmc"

    def __init__(self, alpha=0.01, beta=0.0, eta=4e-6, n_steps=3):

        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.n_steps = n_steps

    def setup(self, model):

        self.model = model
        self.state = clone_parameters(model)
        self.v = {k: torch.empty_like(x) for k, x in self.state.items()}
        self.resample_v()
        self.noise = Normal(0.0, 2 * (self.alpha - self.beta) * self.eta)

    @torch.no_grad()
    def resample_v(self):
        # TODO: Fix resampling?
        for v in self.v.values():
            v.normal_()

    @torch.no_grad()
    def step_parameters(self):
        for name, param in self.model.named_parameters():
            param.copy_(param + self.v[name])

    def step_v(self, x, y):

        self.model.zero_grad()
        (-self.model.log_joint(x, y)).backward()
        parameters = dict(self.model.named_parameters())

        for name, v in self.v.items():
            v.copy_(
                v
                - self.eta * parameters[name].grad
                - self.alpha * v
                + self.noise.sample(v.shape)
            )

    def next_sample(self, batch):

        x, y = batch

        for i in range(self.n_steps):
            self.step_parameters()
            self.step_v(x, y)

        return clone_parameters(self.model)
