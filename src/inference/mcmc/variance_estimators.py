import torch
import torch.nn as nn

from src.inference.mcmc.samplable import ParameterPosterior


class VarianceEstimator(nn.Module):
    
    def setup(self, sampler, inference_module=None):
        pass

    def on_train_epoch_start(self, inference_module):
        pass

    def on_after_grad(self, grad: torch.Tensor):
        pass

    def on_before_next_sample(self, sampler):
        pass

    def estimate(self) -> torch.Tensor:
        raise NotImplementedError


class ConstantEstimator(VarianceEstimator):
    def __init__(self, value=0.0):

        super().__init__()
        self.register_buffer("value",torch.tensor(value))

    def estimate(self):
        return self.value


class AdamEstimator(VarianceEstimator):
    def __init__(self, beta_1=0.9, beta_2=0.999, adj_with_mean=False):

        super().__init__()
        self.register_buffer("beta_1", torch.tensor(beta_1))
        self.register_buffer("beta_2", torch.tensor(beta_2))

        self.adj_with_mean = adj_with_mean

    def setup(self, sampler):
        shape = sampler.samplable.shape
        self.register_buffer("mean_est", torch.zeros(shape))
        self.register_buffer("var_est", torch.zeros(shape))
        self.t = 0
        self.beta_1_pow_t = 1.0
        self.beta_2_pow_t = 1.0

    def on_after_grad(self, grad: torch.Tensor):

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


class WelfordEstimator(nn.Module):

    def __init__(self, shape):
        super().__init__()
        self.n = 0
        self.register_buffer("S_n", torch.zeros(shape))
        self.register_buffer("m_n", torch.zeros(shape))

    def reset(self):

        self.n = 0
        self.S_n.zero_()
        self.m_n.zero_()

    def update(self, value):

        n = self.n + 1

        if n > 1:
            self.S_n.add_(((n - 1) / n) * (value - self.m_n) ** 2)
            self.m_n.mul_((n - 1) / n)
        self.m_n.add_(value / n)

        self.n = n

    def estimate(self):
        return self.S_n / (self.n - 1)


class NoStepException(Exception):
    ...

class NextEpochException(Exception):
    ...

class InterBatchEstimator(VarianceEstimator):

    def __init__(self, n_estimation_steps=10):

        super().__init__()

        self.n_estimation_steps = n_estimation_steps
        self.is_estimating = False

    def setup(self, sampler):

        shape = sampler.samplable.shape
        self.wf_estimator = WelfordEstimator(shape)

    def on_train_epoch_start(self, inference_module):

        self.is_estimating = inference_module.current_epoch % 2 == 0
        if self.is_estimating:
            self.wf_estimator.reset()

    def on_before_next_sample(self, sampler):
        if self.is_estimating:
            sampler.grad_U()
            raise NoStepException

    def on_after_grad(self, grad): 

        if not self.is_estimating:
            return

        self.wf_estimator.update(grad)

        if self.wf_estimator.n == self.n_estimation_steps:
            raise NextEpochException
            
    def estimate(self) -> torch.Tensor:

        # return torch.tensor(0.0)
        if self.wf_estimator.n < 2:
            return torch.tensor(0.0)
        else:
            return self.wf_estimator.estimate()


# class Hybrid:
#     ...


class DummyVarianceEstimator(VarianceEstimator):
    def __init__(self, variance_estimator: VarianceEstimator, use_estimate, constant=0.0):

        super().__init__()
        self.use_estimate = use_estimate
        self.constant = torch.tensor(constant)
        self.wrapped = variance_estimator

    def setup(self, sampler):
        self.wrapped.setup(sampler)

    def estimate(self) -> torch.Tensor:
        if self.use_estimate:
            return self.wrapped.estimate()
        else:
            return self.constant

    def on_train_epoch_start(self, inference_module):
        self.wrapped.on_train_epoch_start(inference_module)

    def on_after_grad(self, grad: torch.Tensor):
        self.wrapped.on_after_grad(grad)

    def on_before_next_sample(self, sampler):
        self.wrapped.on_before_next_sample(sampler)