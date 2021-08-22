from abc import ABC, abstractmethod
import argparse
from src.utils import HPARAM, HyperparameterMixin
import typing
from torch.utils.data.dataset import Dataset
from src.modules import BayesianModel
from typing import Dict, Generator, Generic, Type, TypeVar, Union, Any, Container
import torch
from torch.distributions import Normal
from torch.utils.data import DataLoader
import inspect


@torch.no_grad()
def clone_parameters(model):
    return {k: x.clone() for k, x in model.named_parameters()}


class Sampler(ABC, HyperparameterMixin):

    is_batched: bool

    def sample_generator(
        self, model: BayesianModel, dataset: Dataset
    ) -> Generator[torch.Tensor, None, None]:

        self.setup(model, dataset)
        while True:
            yield self.next_sample()

    @abstractmethod
    def setup(self, model: BayesianModel):
        pass

    @abstractmethod
    def next_sample(self, batch):
        pass


class MetropolisHastings(Sampler):

    is_batched = False

    def __init__(self, step_size=0.01):

        self.step_size = step_size

    def setup(self, model: BayesianModel):

        self.model = model
        self.state = clone_parameters(model)
        self.log_p = None
        
    @torch.no_grad()
    def next_sample(self, batch):

        x, y = batch

        if self.log_p is None:
            self.log_p = self.model.log_joint(x, y)

        for param in self.model.parameters():
            param.copy_(Normal(param, self.step_size).sample())

        new_log_p = self.model.log_joint(x, y)
        log_ratio = new_log_p - self.log_p

        if log_ratio > 0 or torch.bernoulli(log_ratio.exp()):
            self.state = clone_parameters(self.model)
            self.log_p = new_log_p
            return self.state

        else:
            self.model.load_state_dict(self.state, strict=False)
            return None


class Hamiltonian(Sampler):
    """
    M = I for now...
    """

    step_size: HPARAM[float]
    n_steps: HPARAM[int]

    is_batched = False

    def __init__(self, step_size=0.01, n_steps=1) -> None:

        self.step_size = step_size
        self.n_steps = n_steps

    def setup(self, model: BayesianModel):

        self.model = model
        self.state = clone_parameters(model)
        self.momentum = {k: torch.empty_like(x) for k, x in self.state.items()}

    def U(self, x, y):
        return -self.model.log_joint(x, y)

    @torch.no_grad()
    def H(self, x, y):
        result = 0
        for r in self.momentum.values():
            result += torch.dot(r.flatten(), r.flatten()) / 2
        result += self.U(x, y)
        return result

    def resample_momentum(self):
        for r in self.momentum.values():
            r.normal_()

    def step_momentum(self, x, y, half_step=False):
        self.model.zero_grad()
        self.U(x, y).backward()
        parameters = dict(self.model.named_parameters())
        for name, r in self.momentum.items():
            r.copy_(
                r
                - self.step_size * (1 if not half_step else 0.5) * parameters[name].grad
            )

    @torch.no_grad()
    def step_parameters(self):
        for name, param in self.model.named_parameters():
            param.copy_(param + self.step_size * self.momentum[name])

    def next_sample(self, batch):

        x, y = batch

        self.resample_momentum()
        H_current = self.H(x, y)

        self.step_momentum(x, y, half_step=True)

        for _ in range(self.n_steps):

            self.step_parameters()
            self.step_momentum(x, y)

        self.step_momentum(x, y, half_step=True)

        H_proposed = self.H(x, y)
        log_rho = H_proposed - H_current

        if log_rho > 0 or torch.rand(1) < log_rho.exp():
            self.state = clone_parameters(self.model)
            return self.state
        else:
            self.model.load_state_dict(self.state, strict=False)
            return None


class StochasticGradientHamiltonian(Sampler):
    """
    M = I for now... Using diagonal variance estimate
    """

    alpha: HPARAM[float]
    beta: HPARAM[float]
    eta: HPARAM[float]
    n_steps: HPARAM[int]

    is_batched = True

    def __init__(self, alpha=0.01, beta=0.0, eta=4e-6, n_steps=3):

        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.n_steps = n_steps

    def setup(self, model: BayesianModel):

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


class GetSampler(argparse.Action):

    samplers = {
        "mh": MetropolisHastings,
        "hmc": Hamiltonian,
        "sghmc": StochasticGradientHamiltonian,
    }

    default = Hamiltonian

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, self.samplers[values])