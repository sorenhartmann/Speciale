import torch.nn as nn
import torch

class VarianceEstimator(nn.Module):

    def setup(self, shape):
        pass

    def update(self, global_step: int, grad: torch.Tensor):
        pass

    def estimate(self) -> torch.Tensor:
        raise NotImplementedError

class ConstantEstimator(VarianceEstimator):

    def __init__(self, value=0.):

        self.value = torch.tensor(value)

    def estimate(self):
        return self.value
    
class AdamEstimator(VarianceEstimator):

    def __init__(self, beta_1=0.9, beta_2=0.999, adj_with_mean=False):

        super().__init__()
        self.register_buffer("beta_1", torch.tensor(beta_1))
        self.register_buffer("beta_2", torch.tensor(beta_2))

        self.adj_with_mean = adj_with_mean

    def setup(self, shape):

        self.register_buffer("mean_est", torch.zeros(shape))
        self.register_buffer("var_est", torch.zeros(shape))
        self.t = 0
        self.beta_1_pow_t = 1.0
        self.beta_2_pow_t = 1.0

    def update(self, global_step, grad: torch.Tensor):

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
            var_bias_corrected -= mean_bias_corrected**2

        var_bias_corrected.clamp_(min=0)

        return var_bias_corrected 
        # - mean_bias_corrected ** 2

