import torch
import torch.nn as nn

from src.inference.mcmc.samplable import ParameterPosterior


class VarianceEstimator(nn.Module):
    def setup(self, sampler):
        pass

    def update(self, grad: torch.Tensor):
        pass

    def estimate(self) -> torch.Tensor:
        raise NotImplementedError


class ConstantEstimator(VarianceEstimator):
    def __init__(self, value=0.0):

        self.value = torch.tensor(value)

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

    def update(self, grad: torch.Tensor):

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


class NoStepException(Exception):
    ...


class WelfordEstimator(nn.Module):

    def __init__(self):
        super().__init__()

        self.n = 0
        self.initialized_buffers = False

    def reset(self):

        self.n = 0
        if self.initialized_buffers:
            self.S_n.zero_()
            self.m_n.zero_()

    def init_buffers(self, shape):

        self.register_buffer("S_n", torch.zeros(shape))
        self.register_buffer("m_n", torch.zeros(shape))
        self.initialized_buffers = True

    def update(self, value):

        if not self.initialized_buffers:
            self.init_buffers(value.shape)
            
        n = self.n + 1

        if n > 1:
            self.S_n.add_(((n - 1) / n) * (value - self.m_n) ** 2)
            self.m_n.mul_((n - 1) / n)
        self.m_n.add_(value / n)

        self.n = n

    def estimate(self):
        return self.S_n / (self.n-1)

class InterBatchEstimator(VarianceEstimator):

    def __init__(self, n_estimation_steps=10, n_inference_steps=90):

        super().__init__()

        self.n_estimation_steps = n_estimation_steps
        self.n_inference_steps = n_inference_steps
        self.period = n_estimation_steps + n_inference_steps
        self.global_step = 0

    def setup(self, sampler):

        # Method for just calling the grad function, enabling the grad hook, without stepping
        def hook(sampler):
            if self.is_estimating():
                sampler.grad_U()
                raise NoStepException
            else:
                return

        sampler.register_before_next_sample_hook(hook)
        self.wf_estimator = WelfordEstimator()

    def is_estimating(self):
        return self.global_step % self.period < self.n_estimation_steps

    def update(self, grad):

        if not self.is_estimating():
            self.global_step += 1
            return

        if self.global_step % self.period == 0:
            self.wf_estimator.reset()

        self.wf_estimator.update(grad)
        self.global_step += 1

    def estimate(self) -> torch.Tensor:
        if self.wf_estimator.n < 2:
            return torch.tensor(0.0)
        else:
            return self.wf_estimator.estimate()


# class Hybrid:
#     ...
