import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleDict, Parameter
from .priors import NormalPrior, Prior, ScaleMixturePrior
from typing import Dict, Union
import math


class BayesianModule(Module):

    priors: ModuleDict

    def log_prior(self):
        return sum(
            prior.log_prob(getattr(self, name)).sum()
            for name, prior in self.priors.items()
        )

    @classmethod
    def from_freq_module(
        cls, module, priors: Union[Dict[str, Prior], Prior]
    ) -> "BayesianModule":
        raise NotImplementedError


class BayesianLinear(BayesianModule):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        priors: Union[Dict[str, Prior], Prior] = None,
    ):

        super().__init__()
        if priors is None:
            priors = {}

        if isinstance(priors, Prior):
            _default_prior = priors
            priors = {}
        else:
            _default_prior = NormalPrior()

        self.priors = ModuleDict()

        self.priors["weight"] = priors.get("weight", _default_prior)
        self.priors["bias"] = priors.get("bias", _default_prior)

        self.weight = Parameter(torch.empty((out_features, in_features)))

        if bias:
            self.bias = Parameter(torch.empty((out_features)))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self) -> None:

        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):

        if self.weight.dim() == 2:
            return F.linear(x, self.weight, self.bias)

        elif self.weight.dim() == 3:

            assert x.shape[0] == self.weight.shape[0]
            out = torch.matmul(
                self.weight.unsqueeze(1),  # Add dimension for broadcast over batch
                x.unsqueeze(-1),  # Add dimension for mv product
            )
            out.squeeze_(-1)  # Squeeze out mv dimension
            if self.bias is not None:
                out.add_(
                    self.bias.unsqueeze(1)
                )  # Add dimension for broadcast over batch

            return out
        else:
            raise ValueError

    @classmethod
    def from_freq_module(cls, module, priors: Dict[str, Prior] = None):

        has_bias = module.bias is not None

        result = cls(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=has_bias,
            priors=priors,
        )

        with torch.no_grad():
            result.weight.copy_(module.weight)
            if has_bias:
                result.bias.copy_(module.bias)

        return result
