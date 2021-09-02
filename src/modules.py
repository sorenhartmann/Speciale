from abc import ABC, abstractmethod
from functools import cached_property
from itertools import accumulate
import math
from src.samplers import Samplable

from src.utils import HyperparameterMixin, pairwise
from torch import nn
import torch
from torch.distributions import Gamma, Normal


class BayesianMixin(ABC):

    def setup_prior(self):
        pass

    @abstractmethod
    def log_prior(self):
        pass

class BayesianModel(nn.Module, HyperparameterMixin, Samplable):
    
    @abstractmethod
    def observation_model(self, x):
        """Returns p(y | x, theta)"""
        mu = self.forward(x)
        return torch.distributions.Normal(mu, 1.0)

    def log_prior(self):
        """Returns p(theta)"""
        return sum(
            m.log_prior() for m in self.modules() if isinstance(m, BayesianMixin)
        )

    def log_likelihood(self, x: torch.FloatTensor, y: torch.FloatTensor):
        """Returns log p(y | x, theta)"""
        return self.observation_model(x).log_prob(y)

    def prop_log_p(self, x, y) -> torch.Tensor:
        return self.log_prior() + self.log_likelihood(x, y).sum()

    def grad_prop_log_p(self, x, y):
        self.zero_grad()
        self.prop_log_p(x, y).backward()
        return self.state_grad

    @cached_property
    def param_shapes(self):
        return {k: x.shape for k, x in self.named_parameters()}

    @cached_property
    def flat_index_pairs(self):
        indices = accumulate(
            self.param_shapes.values(), lambda x, y: x + math.prod(y), initial=0
        )
        return list(pairwise(indices))

    @property
    def state(self) -> torch.Tensor: 
        return torch.cat([x.detach().flatten() for x in self.parameters()])

    @state.setter
    def state(self, value):
        self.load_state_dict(
            {
                k: value[a:b].view(shape)
                for (k, shape), (a, b) in zip(
                    self.param_shapes.items(), self.flat_index_pairs
                )
            },
            strict=False
        )

    @property
    def state_grad(self) -> torch.Tensor:
        return torch.cat([x.grad.flatten() for x in self.parameters()])

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
