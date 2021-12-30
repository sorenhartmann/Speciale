from typing import TYPE_CHECKING, Optional, Union

import torch
from torch import Tensor

from src.inference.mcmc.samplable import Samplable
from src.inference.mcmc.variance_estimators import (
    ConstantEstimator,
    ExpWeightedEstimator,
    VarianceEstimator,
)

if TYPE_CHECKING:
    from src.inference.mcmc import MCMCInference


class Sampler(torch.nn.Module):

    samplable: Samplable

    def on_train_epoch_start(self, inference_module: "MCMCInference") -> None:
        """Only called in case of inference sampling"""

    def setup(self, samplable: Samplable) -> "Sampler":
        raise NotImplementedError

    def next_sample(self, return_sample: bool = True) -> Optional[Tensor]:
        raise NotImplementedError


class HamiltonianMixin:

    samplable: Samplable
    momentum: Tensor

    def U(self) -> Tensor:
        return -self.samplable.prop_log_p()

    def grad_U(self) -> Tensor:
        grad = self.samplable.grad_prop_log_p()
        return -grad

    def H(self) -> Tensor:
        return self.U() + self.momentum.square().sum() / 2


class HMC(Sampler, HamiltonianMixin):
    """
    M = I for now...
    """

    def __init__(self, step_size: float = 0.01, n_steps: int = 1) -> None:

        super().__init__()

        self.register_buffer("step_size", torch.tensor(step_size))
        self.step_size: Tensor
        self.n_steps = n_steps

    def setup(self, samplable: Samplable) -> "HMC":

        self.samplable = samplable
        self.momentum = torch.empty_like(self.samplable.state)

        return self

    def resample_momentum(self) -> None:
        self.momentum.normal_()

    def step_momentum(self, half_step: bool = False) -> None:

        self.momentum.copy_(
            self.momentum
            - self.step_size * (1.0 if not half_step else 0.5) * self.grad_U()
        )

    def step_parameters(self) -> None:

        self.samplable.state = self.samplable.state + self.step_size * self.momentum

    def next_sample(self, return_sample: bool = True) -> Optional[Tensor]:

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

        return None


class HMCNoMH(HMC):
    def next_sample(self, return_sample: bool = True) -> Optional[Tensor]:

        self.resample_momentum()
        self.step_momentum(half_step=True)
        for i in range(self.n_steps):
            self.step_parameters()
            self.step_momentum(half_step=(i == self.n_steps - 1))

        if return_sample:
            return self.samplable.state.clone()
        else:
            return None


import torch


class SGHMC(Sampler, HamiltonianMixin):

    _before_next_sample_hook = None

    def __init__(
        self,
        alpha: float = 1e-2,
        beta: float = 0.0,
        lr: float = 2e-6,
        resample_momentum_every: int = 50,
    ):
        super().__init__()

        self.register_buffer("alpha", torch.tensor(alpha))
        self.alpha: Tensor
        self.register_buffer("beta", torch.tensor(beta))
        self.beta: Tensor
        self.register_buffer("lr", torch.tensor(lr))
        self.lr: Tensor

        self.resample_momentum_every = resample_momentum_every

        if self.resample_momentum_every:
            self.steps_until_momentum_resample = self.resample_momentum_every
        else:
            self.steps_until_momentum_resample = -1

    def err_std(self) -> Tensor:
        return torch.sqrt(2 * (self.alpha - self.beta) * self.lr)

    def setup(self, samplable: Samplable) -> "SGHMC":

        self.samplable = samplable
        self.register_buffer("nu", torch.zeros_like(self.samplable.state))
        self.nu: Tensor
        self.resample_nu()

        return self

    def step_parameters(self) -> None:

        self.samplable.state = self.samplable.state + self.nu

    def resample_nu(self) -> None:

        self.nu.normal_()
        self.nu.mul_(self.lr.sqrt())

    def step_nu(self) -> None:

        self.nu.add_(
            -self.lr * self.grad_U()
            - self.alpha * self.nu
            + torch.randn_like(self.nu) * self.err_std
        )

    def next_sample(self, return_sample: bool = True) -> Optional[Tensor]:

        if self.resample_momentum_every:
            self.steps_until_momentum_resample -= 1

        if self.steps_until_momentum_resample == 0:
            self.resample_nu()
            self.steps_until_momentum_resample = self.resample_momentum_every

        self.step_nu()
        self.step_parameters()

        if return_sample:
            return self.samplable.state.clone()
        else:
            return None


class SGHMCWithVarianceEstimator(SGHMC, HamiltonianMixin):
    def __init__(
        self,
        alpha: float = 1e-2,
        lr: float = 2e-6,
        variance_estimator: Union[float, VarianceEstimator] = None,
        resample_momentum_every: int = 50,
        estimation_margin: float = 10.0,
        rescale_mass_every: int = 100,
    ) -> None:

        torch.nn.Module.__init__(self)

        self.register_buffer("alpha", torch.tensor(alpha))
        self.alpha: Tensor
        self.register_buffer("lr_0", torch.tensor(lr))
        self.lr_0: Tensor
        self.register_buffer("lr", torch.tensor(lr))
        self.lr: Tensor

        if variance_estimator is None:
            variance_estimator = ExpWeightedEstimator()
        elif isinstance(variance_estimator, float):
            variance_estimator = ConstantEstimator(variance_estimator)
        # variance_estimator: VarianceEstimator

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
    def err_std(self) -> Tensor:
        variance_estimate = self.variance_estimator.estimate()
        beta = self.lr * variance_estimate / 2
        return torch.sqrt(2 * (self.alpha - beta).clamp(min=0) * self.lr)

    def setup(self, samplable: Samplable) -> "SGHMCWithVarianceEstimator":

        self.samplable = samplable
        self.register_buffer("nu", torch.zeros_like(self.samplable.state))
        self.nu: Tensor
        self.register_buffer("mass_factor", torch.ones_like(self.samplable.state))
        self.mass_factor: Tensor
        self.variance_estimator.setup(self)
        self.resample_nu()

        return self

    def rescale_mass(self, variance_estimate: Tensor) -> None:

        lower_bound = (
            self.estimation_margin * variance_estimate * self.lr_0 / (2 * self.alpha)
        )
        new_mass_factor = lower_bound.clamp(min=1)

        self.lr = self.lr_0 / new_mass_factor
        self.nu = self.nu * self.mass_factor / new_mass_factor
        self.mass_factor = new_mass_factor

    def grad_U(self) -> Tensor:
        grad = super().grad_U()
        self.variance_estimator.on_after_grad(grad)
        return grad

    def next_sample(self, return_sample: bool = True) -> Optional[Tensor]:

        self.variance_estimator.on_before_next_sample(self)

        if self.rescale_mass_every:
            self.steps_until_mass_rescaling -= 1

        if self.steps_until_mass_rescaling == 0:
            variance_estimate = self.variance_estimator.estimate()
            self.rescale_mass(variance_estimate)
            self.steps_until_mass_rescaling = self.rescale_mass_every

        return super().next_sample(return_sample)

    def on_train_epoch_start(self, inference_module: "MCMCInference") -> None:
        self.variance_estimator.on_train_epoch_start(inference_module)
