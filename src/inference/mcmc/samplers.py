from abc import ABC, abstractmethod
from typing import Union

import torch
from torch.distributions import Normal

from src.inference.mcmc.samplable import Samplable
from src.inference.mcmc.variance_estimators import (
    ConstantEstimator,
    InterBatchEstimator,
    VarianceEstimator,
)


class Sampler(torch.nn.Module):
    def on_train_epoch_start(self, inference_module):
        """Only called in case of inference sampling"""

    def setup(self, samplable: Samplable):
        raise NotImplementedError

    def next_sample(self):
        raise NotImplementedError


class MetropolisHastings(Sampler):
    def __init__(self, step_size=0.01):

        super().__init__()
        self.register_buffer("step_size", torch.tensor(0.01))

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


class HMC(Sampler, HamiltonianMixin):
    """
    M = I for now...
    """

    def __init__(self, step_size=0.01, n_steps=1) -> None:

        super().__init__()

        self.register_buffer("step_size", torch.tensor(step_size))
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

    def next_sample(self, return_sample: bool = True):

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
            if return_sample:
                return self.samplable.state.clone()

        else:
            # Rejected
            self.samplable.state = initial_state
            if return_sample:
                return initial_state


class HMCNoMH(HMC):
    def next_sample(self):

        self.resample_momentum()
        self.step_momentum(half_step=True)
        for i in range(self.n_steps):
            self.step_parameters()
            self.step_momentum(half_step=(i == self.n_steps - 1))

        return self.samplable.state.clone()


import torch
import torch.nn as nn


class SGHMC(Sampler, HamiltonianMixin):

    _before_next_sample_hook = None

    def __init__(
        self,
        alpha: float = 1e-2,
        beta: float = 0.0,
        lr: float = 0.2e-5,
        resample_momentum_every: int = 50,
    ):
        super().__init__()

        self.alpha = torch.tensor(alpha)
        self.beta = torch.tensor(beta)
        self.lr = torch.tensor(lr)

        self.err_std = torch.sqrt(2 * (self.alpha - self.beta) * self.lr)

        self.resample_momentum_every = resample_momentum_every

        if self.resample_momentum_every:
            self.steps_until_momentum_resample = self.resample_momentum_every
        else:
            self.steps_until_momentum_resample = -1

    def setup(self, samplable: Samplable):

        self.samplable = samplable
        self.register_buffer("nu", torch.zeros_like(self.samplable.state))

        return self

    def step_parameters(self):

        self.samplable.state = self.samplable.state + self.nu

    def resample_nu(self):

        self.nu.normal_()
        self.nu.mul_(self.lr.sqrt())

    def step_nu(self):

        self.nu.add_(
            -self.lr * self.grad_U()
            - self.alpha * self.nu
            + torch.randn_like(self.nu) * self.err_std
        )

    def next_sample(self, return_sample: bool = True):

        if self.resample_momentum_every:
            self.steps_until_momentum_resample -= 1

        if self.steps_until_momentum_resample == 0:
            self.resample_nu()
            self.steps_until_momentum_resample = self.resample_momentum_every

        self.step_nu()
        self.step_parameters()

        if return_sample:
            return self.samplable.state.clone()


# def sghmc_original_parameterization(
#     step_size: int, B: float, M: float, C: float, n_steps: int = 1
# ):

#     lr = step_size ** 2 / M
#     alpha = step_size / M * C
#     beta = step_size / M * B

#     return SGHMC(alpha, beta, lr, n_steps, resample_momentum=True)


class SGHMCWithVarianceEstimator(SGHMC, HamiltonianMixin):
    def __init__(
        self,
        alpha: float = 1e-2,
        lr: float = 2e-6,
        variance_estimator: Union[float, VarianceEstimator] = None,
        resample_momentum_every: int = 50,
        estimation_margin=10,
        rescale_mass_every : int = 100
    ):

        torch.nn.Module.__init__(self)

        self.register_buffer("alpha", torch.tensor(alpha))
        self.register_buffer("lr_0", torch.tensor(lr))
        self.lr = self.lr_0

        if variance_estimator is None:
            variance_estimator = InterBatchEstimator()
        elif type(variance_estimator) is float:
            variance_estimator = ConstantEstimator(variance_estimator)

        self.variance_estimator = variance_estimator
        self.resample_momentum_every = resample_momentum_every
        self.estimation_margin = estimation_margin
        self.rescale_mass_every = rescale_mass_every

        if self.resample_momentum_every:
            self.steps_until_momentum_resample = self.resample_momentum_every
        else:
            self.steps_until_momentum_resample = -1

        if self.rescale_mass_every:
            self.steps_until_mass_rescaling = self.rescale_mass_every
        else:
            self.steps_until_mass_rescaling = -1

    @property
    def err_std(self):
        variance_estimate = self.variance_estimator.estimate()
        beta = self.lr * variance_estimate / 2
        return torch.sqrt(2 * (self.alpha - beta).clamp(min=0) * self.lr)

    def rescale_mass(self, variance_estimate):

        lower_bound = (
            self.estimation_margin * variance_estimate * self.lr_0 / (2 * self.alpha)
        )
        new_mass_factor = lower_bound.clamp(min=1)
        self.lr = self.lr_0 / new_mass_factor
        self.nu = self.nu * self.mass_factor / new_mass_factor
        self.mass_factor = new_mass_factor

    def grad_U(self):
        grad = super().grad_U()
        self.variance_estimator.on_after_grad(grad)
        return grad

    def setup(self, samplable: Samplable):

        self.samplable = samplable
        self.register_buffer("nu", torch.zeros_like(self.samplable.state))
        self.register_buffer("mass_factor", torch.ones_like(self.samplable.state))
        self.variance_estimator.setup(self)
        self.resample_nu()

        return self

    def next_sample(self, return_sample: bool = True):

        self.variance_estimator.on_before_next_sample(self)

        if self.rescale_mass_every:
            self.steps_until_mass_rescaling -= 1

        if self.steps_until_mass_rescaling == 0:
            variance_estimate = self.variance_estimator.estimate()
            self.rescale_mass(variance_estimate)
            self.steps_until_mass_rescaling = self.rescale_mass_every
        
        return super().next_sample(return_sample)

    def on_train_epoch_start(self, inference_module):

        self.variance_estimator.on_train_epoch_start(inference_module)
