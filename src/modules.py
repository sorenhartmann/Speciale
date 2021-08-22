from abc import ABC, abstractmethod
from src.utils import HyperparameterMixin
from torch import nn
import torch
from torch.distributions import Gamma, Normal

class BayesianMixin(ABC):
    def setup_prior(self):
        pass

    @abstractmethod
    def log_prior(self):
        pass

class BayesianModel(nn.Module, HyperparameterMixin):

    def log_prior(self):
        return sum(
            m.log_prior() for m in self.modules() if isinstance(m, BayesianMixin)
        )

    @abstractmethod
    def log_likelihood(self, x, y):
        pass

    def log_joint(self, x, y):
        return self.log_prior() + self.log_likelihood(x, y).sum()


class BayesianLinearKnownPrecision(BayesianMixin, nn.Linear):

    weight_precision: torch.Tensor
    bias_precision: torch.Tensor

    def setup_prior(self, precision):

        self.register_buffer("weight_precision", torch.tensor(precision))
        self.register_buffer("bias_precision", torch.tensor(precision))

        return self

    def log_prior(self):
        weight_param_d = Normal(0, 1 / self.weight_precision)
        bias_param_d = Normal(0, 1 / self.bias_precision)
        return (
            weight_param_d.log_prob(self.weight).sum() 
            + bias_param_d.log_prob(self.bias).sum()
        )


class BayesianLinear(BayesianMixin, nn.Linear):

    weight_precision: nn.Parameter
    bias_precision: nn.Parameter
    alpha: torch.Tensor
    beta: torch.Tensor

    def setup_prior(self, alpha: float, beta: float):

        self.register_parameter("weight_precision", nn.Parameter(torch.tensor(1.0)))
        self.register_parameter("bias_precision", nn.Parameter(torch.tensor(1.0)))
        self.register_buffer("alpha", torch.tensor(alpha))
        self.register_buffer("beta", torch.tensor(beta))

        return self

    def log_prior(self):

        weight_precision_d = Gamma(self.alpha, self.beta)
        weight_d = Normal(0, 1 / self.weight_precision)
        bias_precision_d = Gamma(self.alpha, self.beta)
        bias_d = Normal(0, 1 / self.bias_precision)

        return (
            weight_precision_d.log_prob(self.weight_precision)
            + bias_precision_d.log_prob(self.bias_precision)
            + weight_d.log_prob(self.weight).sum()
            + bias_d.log_prob(self.bias).sum()
        )
