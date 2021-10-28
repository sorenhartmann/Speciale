from abc import ABC, abstractmethod
from typing import Union

import torch
from torch.distributions import Normal

from src.inference.mcmc.samplable import Samplable
from src.inference.mcmc.variance_estimators import (ConstantEstimator,
                                               VarianceEstimator)


class Sampler(torch.nn.Module):

    is_batched: bool

    def on_train_epoch_start(self, inference_module):
        """Only called in case of inference sampling"""

    def setup(self, samplable: Samplable):
        raise NotImplementedError

    def next_sample(self):
        raise NotImplementedError

class MetropolisHastings(Sampler):

    is_batched = False

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

    is_batched = False

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

    is_batched = True

    _before_next_sample_hook = None

    def __init__(
        self,
        alpha: float = 1e-2,
        beta: float = 0.0,
        lr: float = 0.2e-5,
        n_steps: int = 1,
        resample_momentum: bool = False,
    ):
        super().__init__()

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

    def next_sample(self, return_sample: bool = True):

        if self.resample_momentum:
            self.resample_nu()

        for i in range(self.n_steps):
            self.step_nu()
            self.step_parameters()

        if return_sample:
            return self.samplable.state.clone()


def sghmc_original_parameterization(
    step_size: int, B: float, M: float, C: float, n_steps: int = 1
):

    lr = step_size ** 2 / M
    alpha = step_size / M * C
    beta = step_size / M * B

    return SGHMC(alpha, beta, lr, n_steps, resample_momentum=True)


class SGHMCWithVarianceEstimator(SGHMC, HamiltonianMixin):
    def __init__(
        self,
        alpha: float = 1e-2,
        lr: float = 0.2e-5,
        variance_estimator: Union[float, VarianceEstimator] = 0.0,
        n_steps: int = 1,
        resample_momentum: bool = False,
    ):

        torch.nn.Module.__init__(self)

        self.register_buffer("alpha", torch.tensor(alpha))
        self.register_buffer("lr", torch.tensor(lr))

        if type(variance_estimator) is float:
            self.variance_estimator = ConstantEstimator(float)
        else:
            self.variance_estimator = variance_estimator

        self.n_steps = n_steps
        self.resample_momentum = resample_momentum

    @property
    def beta(self):
        var_estimate = self.variance_estimator.estimate()
        return 2 * var_estimate * self.lr

    @property
    def err_std(self):
        beta = self.beta
        alpha = self.alpha.clamp(min=2 * beta)
        return torch.sqrt(2 * (alpha - beta) * self.lr)

    def grad_U(self):
        grad = super().grad_U()
        self.variance_estimator.on_after_grad(grad)
        return grad

    def setup(self, samplable: Samplable):
        super().setup(samplable)
        self.variance_estimator.setup(self) 
        return self

    def next_sample(self, return_sample: bool = True):
        self.variance_estimator.on_before_next_sample(self)
        return super().next_sample(return_sample)

    def on_train_epoch_start(self, inference_module):
        self.variance_estimator.on_train_epoch_start(inference_module)