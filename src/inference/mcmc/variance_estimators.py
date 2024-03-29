from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch._C import Size

if TYPE_CHECKING:
    from .mcmc import MCMCInference
    from .samplers import SGHMCWithVarianceEstimator


class VarianceEstimator(nn.Module):
    def setup(
        self,
        sampler: "SGHMCWithVarianceEstimator",
        inference_module: Optional["MCMCInference"] = None,
    ) -> None:
        pass

    def on_train_epoch_start(self, inference_module: Optional["MCMCInference"]) -> None:
        pass

    def on_after_grad(self, grad: torch.Tensor) -> None:
        pass

    def on_before_next_sample(self, sampler: "SGHMCWithVarianceEstimator") -> None:
        pass

    def estimate(self) -> Tensor:
        raise NotImplementedError


class ConstantEstimator(VarianceEstimator):
    def __init__(self, value: float = 0.0) -> None:

        super().__init__()
        self.register_buffer("value", torch.tensor(value))
        self.value: Tensor

    def estimate(self) -> Tensor:
        return self.value


class AdamEstimator(VarianceEstimator):
    def __init__(
        self,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        adj_with_mean: bool = False,
    ):

        super().__init__()
        self.register_buffer("beta_1", torch.tensor(beta_1))
        self.beta_1: Tensor
        self.register_buffer("beta_2", torch.tensor(beta_2))
        self.beta_2: Tensor

        self.adj_with_mean = adj_with_mean

    def setup(
        self,
        sampler: "SGHMCWithVarianceEstimator",
        inference_module: Optional["MCMCInference"] = None,
    ) -> None:
        shape = sampler.samplable.shape
        self.register_buffer("mean_est", torch.zeros(shape))
        self.mean_est: Tensor
        self.register_buffer("var_est", torch.zeros(shape))
        self.var_est: Tensor
        self.t = 0
        self.beta_1_pow_t = torch.tensor(1.0)
        self.beta_2_pow_t = torch.tensor(1.0)

    def on_after_grad(self, grad: torch.Tensor) -> None:

        self.t += 1
        self.beta_1_pow_t = self.beta_1_pow_t * self.beta_1
        self.beta_2_pow_t = self.beta_2_pow_t * self.beta_2

        self.mean_est.mul_(self.beta_1)
        self.mean_est.addcmul_(1 - self.beta_1, grad)

        self.var_est.mul_(self.beta_2)
        self.var_est.addcmul_(1 - self.beta_2, grad ** 2)

    def estimate(self) -> torch.Tensor:

        var_bias_corrected = self.var_est / (1 - self.beta_2_pow_t)

        if self.adj_with_mean:
            mean_bias_corrected = self.mean_est / (1 - self.beta_1_pow_t)
            var_bias_corrected -= mean_bias_corrected ** 2

        var_bias_corrected.clamp_(min=0)

        return var_bias_corrected


class ExpWeightedEstimator(VarianceEstimator):
    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha

    def setup(
        self,
        sampler: "SGHMCWithVarianceEstimator",
        inference_module: Optional["MCMCInference"] = None,
    ) -> None:
        self.register_buffer("mean", torch.zeros(sampler.samplable.shape))
        self.mean: Tensor
        self.register_buffer("variance", torch.ones(sampler.samplable.shape))
        self.variance: Tensor

    def on_after_grad(self, grad: torch.Tensor) -> None:
        diff = grad - self.mean
        incr = self.alpha * diff
        self.mean = self.mean + incr
        self.variance = (1 - self.alpha) * (self.variance + diff * incr)

    def estimate(self) -> torch.Tensor:
        return self.variance


class WelfordEstimator(nn.Module):
    def __init__(self, shape: Size):
        super().__init__()
        self.n = 0
        self.register_buffer("S_n", torch.zeros(shape))
        self.S_n: Tensor
        self.register_buffer("m_n", torch.zeros(shape))
        self.m_n: Tensor

    def reset(self) -> None:

        self.n = 0
        self.S_n.zero_()
        self.m_n.zero_()

    def update(self, value: Tensor) -> None:

        n = self.n + 1

        if n > 1:
            self.S_n.add_(((n - 1) / n) * (value - self.m_n) ** 2)
            self.m_n.mul_((n - 1) / n)
        self.m_n.add_(value / n)

        self.n = n

    def estimate(self) -> Tensor:
        return self.S_n / (self.n - 1)


class NoStepException(Exception):
    ...


class NextEpochException(Exception):
    ...


class InterBatchEstimator(VarianceEstimator):
    def __init__(self, n_estimation_steps: int = 10) -> None:

        super().__init__()

        self.n_estimation_steps = n_estimation_steps
        self.is_estimating = False

    def setup(
        self,
        sampler: "SGHMCWithVarianceEstimator",
        inference_module: Optional["MCMCInference"] = None,
    ) -> None:

        shape = sampler.samplable.shape
        self.wf_estimator = WelfordEstimator(shape)

    def on_train_epoch_start(self, inference_module: Optional["MCMCInference"]) -> None:

        assert inference_module is not None

        self.is_estimating = inference_module.current_epoch % 2 == 0
        if self.is_estimating:
            self.wf_estimator.reset()

    def on_before_next_sample(self, sampler: "SGHMCWithVarianceEstimator") -> None:

        if self.is_estimating:
            sampler.grad_U()
            raise NoStepException

    def on_after_grad(self, grad: Tensor) -> None:

        if not self.is_estimating:
            return

        self.wf_estimator.update(grad)

        if self.wf_estimator.n == self.n_estimation_steps:
            raise NextEpochException

    def estimate(self) -> torch.Tensor:

        if self.wf_estimator.n < 2:
            return torch.tensor(0.0)
        else:
            return self.wf_estimator.estimate()


class DummyVarianceEstimator(VarianceEstimator):
    def __init__(
        self,
        variance_estimator: VarianceEstimator,
        use_estimate: bool,
        constant: float = 0.0,
    ) -> None:

        super().__init__()
        self.use_estimate = use_estimate
        self.constant = torch.tensor(constant)
        self.wrapped = variance_estimator

    def setup(
        self,
        sampler: "SGHMCWithVarianceEstimator",
        inference_module: Optional["MCMCInference"] = None,
    ) -> None:
        self.wrapped.setup(sampler)

    def estimate(self) -> torch.Tensor:
        if self.use_estimate:
            return self.wrapped.estimate()
        else:
            return self.constant

    def on_train_epoch_start(
        self, inference_module: Optional["MCMCInference"] = None
    ) -> None:
        self.wrapped.on_train_epoch_start(inference_module)

    def on_after_grad(self, grad: torch.Tensor) -> None:
        self.wrapped.on_after_grad(grad)

    def on_before_next_sample(self, sampler: "SGHMCWithVarianceEstimator") -> None:
        self.wrapped.on_before_next_sample(sampler)
