from abc import ABC, abstractmethod
from torch import nn
import torch
from torch.distributions import Gamma, Normal


class BayesianMixin(ABC):
    
    def setup_prior(self):
        pass

    @abstractmethod
    def log_prior(self):
        pass

class BayesianModel(nn.Module):
    
    def log_prior(self):
        return sum(
            m.log_prior() for m in self.modules() if isinstance(m, BayesianMixin)
        )

    @abstractmethod
    def log_likelihood(self, x, y):
        pass

    def log_joint(self, x, y):
        return self.log_prior() + self.log_likelihood(x, y)

class BayesianLinearKnownPrecision(BayesianMixin, nn.Linear):

    precision: torch.Tensor

    def setup_prior(self, precision):
        self.register_buffer("precision", torch.tensor(precision))
        return self

    def log_prior(self):
        param_d = Normal(0, self.precision)
        return param_d.log_prob(self.weight).sum() + param_d.log_prob(self.bias).sum()


class BayesianLinear(BayesianMixin, nn.Linear):

    precision: torch.Tensor
    alpha: torch.Tensor
    beta: nn.Parameter

    def setup_prior(self, alpha, beta):

        self.register_parameter("precision", nn.Parameter(torch.tensor(1.0)))
        self.register_buffer("alpha", torch.tensor(alpha))
        self.register_buffer("beta", torch.tensor(beta))

        return self

    def log_prior(self):

        precision_d = Gamma(self.alpha, self.beta)
        param_d = Normal(0, self.precision)

        return (
            precision_d.log_prob(self.precision)
            + param_d.log_prob(self.weight).sum()
            + param_d.log_prob(self.bias).sum()
        )


