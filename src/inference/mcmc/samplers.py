from abc import ABC, abstractmethod, abstractproperty

import torch
from torch.distributions import Normal


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


class Sampler(ABC):

    is_batched: bool

    @abstractmethod
    def setup(self, samplable: Samplable):
        pass

    @abstractmethod
    def next_sample(self):
        pass


class MetropolisHastings(Sampler):

    is_batched = False

    def __init__(self, step_size=0.01):

        self.step_size = step_size

    def setup(self, samplable):

        self.samplable = samplable
        self.log_p = None

    @torch.no_grad()
    def next_sample(self, return_sample=True):

        if self.log_p is None:
            self.log_p = self.samplable.prop_log_p()

        self.samplable.state
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


class HamiltonianMixin:
    def U(self):
        return -self.samplable.prop_log_p()

    def grad_U(self):
        grad = self.samplable.grad_prop_log_p()
        return -grad

    def H(
        self,
    ):
        return self.U() + self.momentum.square().sum() / 2


class Hamiltonian(Sampler, HamiltonianMixin):
    """
    M = I for now...
    """

    is_batched = False

    def __init__(self, step_size=0.01, n_steps=1) -> None:

        self.step_size = step_size
        self.n_steps = n_steps

    def setup(self, samplable: Samplable):

        self.samplable = samplable
        self.momentum = torch.empty_like(self.samplable.state)

        return self

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


class StochasticGradientHamiltonian(Sampler, HamiltonianMixin):

    is_batched = True

    def __init__(
        self, alpha=1e-2, beta=0.0, lr=0.2e-5, n_steps=1, resample_momentum=False
    ):

        self.alpha = torch.tensor(alpha)
        self.beta = torch.tensor(beta)
        self.lr = torch.tensor(lr)

        self.err_std = torch.sqrt(2 * (self.alpha - self.beta) * self.lr)

        self.n_steps = n_steps

        self.resample_momentum = resample_momentum

    def setup(self, samplable: Samplable):

        self.samplable = samplable
        self.nu = torch.zeros_like(self.samplable.state)
        self.resample_nu()
        return self

    def step_parameters(self):

        self.samplable.state = self.samplable.state + self.nu

    def resample_nu(self):
        self.nu.normal_(0, self.lr.sqrt())

    def step_nu(self):

        self.nu.add_(
            -self.lr * self.grad_U()
            - self.alpha * self.nu
            + torch.randn_like(self.nu) * self.err_std
        )

    def next_sample(self, return_sample=True):

        if self.resample_momentum:
            self.resample_nu()

        for i in range(self.n_steps):
            self.step_nu()
            self.step_parameters()

        if return_sample:
            return self.samplable.state.clone()


def sghmc_original_parameterization(step_size, B, M, C, n_steps=1):

    lr = step_size ** 2 / M
    alpha = step_size / M * C
    beta = step_size / M * B

    return StochasticGradientHamiltonian(
        alpha, beta, lr, n_steps, resample_momentum=True
    )
