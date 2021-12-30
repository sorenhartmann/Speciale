import math
from typing import Dict, Optional, Tuple, Union, cast

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Conv2d, Linear, Module, ModuleDict, Parameter

from .priors import Prior


class BayesianModule(Module):

    priors: ModuleDict

    def log_prior(self) -> Tensor:

        return sum(
            (
                prior.log_prob(getattr(self, name)).sum()
                for name, prior in self.priors.items()
            ),
            torch.tensor(0),
        )

    @classmethod
    def from_freq_module(
        cls, module: Module, priors: Dict[str, Prior]
    ) -> "BayesianModule":
        raise NotImplementedError


class BayesianLinear(BayesianModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        priors: Dict[str, Prior] = None,
    ) -> None:

        super().__init__()

        assert priors is not None

        self.priors = ModuleDict()
        self.priors["weight"] = priors["weight"]

        self.weight = Parameter(torch.empty((out_features, in_features)))

        self.bias: Optional[Parameter]
        if bias:
            self.bias = Parameter(torch.empty((out_features)))
            self.priors["bias"] = priors["bias"]
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self) -> None:

        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias)

    @classmethod
    def from_freq_module(
        cls, module: Module, priors: Dict[str, Prior]
    ) -> "BayesianLinear":

        assert isinstance(module, Linear)

        has_bias = module.bias is not None

        result = cls(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=has_bias,
            priors=priors,
        )

        with torch.no_grad():
            result.weight.copy_(module.weight)
            if result.bias is not None:
                result.bias.copy_(module.bias)

        return result


class BayesianScaleShift(BayesianModule):
    def __init__(self, dim: int, dim_size: int, priors: Dict[str, Prior]) -> None:

        super().__init__()

        self.dim = dim
        self.register_parameter("scale", torch.nn.Parameter(torch.empty(dim_size)))
        self.scale: Tensor
        self.register_parameter("shift", torch.nn.Parameter(torch.empty(dim_size)))
        self.shift: Tensor

        self.priors = ModuleDict()
        self.priors["scale"] = priors["scale"]
        self.priors["shift"] = priors["shift"]

    def forward(self, x: Tensor) -> Tensor:
        in_dim = len(x.shape)
        view_shape = (1,) * self.dim + (-1,) + (1,) * (in_dim - 1 - self.dim)
        return self.scale.view(view_shape) * x + self.shift.view(view_shape)

    @classmethod
    def from_freq_module(
        cls, module: Module, priors: Dict[str, Prior]
    ) -> "BayesianScaleShift":

        assert isinstance(module, torch.nn.BatchNorm2d)

        running_var = (
            module.running_var if module.running_var is not None else torch.tensor(1.0)
        )
        running_mean = (
            module.running_mean
            if module.running_mean is not None
            else torch.tensor(0.0)
        )

        with torch.no_grad():
            scale = 1 / torch.sqrt(running_var + module.eps) * module.weight
            shift = -running_mean * scale
            if module.bias is not None:
                shift = shift + module.bias

        dim = 1

        result = cls(dim, len(scale), priors)
        with torch.no_grad():
            result.scale.copy_(scale)
            result.shift.copy_(shift)

        return result


_2_TUPLE_OR_INT = Union[int, Tuple[int, int]]


class BayesianConv2d(BayesianModule, Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _2_TUPLE_OR_INT,
        stride: _2_TUPLE_OR_INT = 1,
        padding: _2_TUPLE_OR_INT = 0,
        dilation: _2_TUPLE_OR_INT = 1,
        groups: int = 1,
        bias: bool = True,
        priors: Dict[str, Prior] = None,
    ):

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        assert priors is not None

        self.priors = ModuleDict()
        self.priors["weight"] = priors["weight"]
        if bias:
            self.priors["bias"] = priors["bias"]

    @classmethod
    def from_freq_module(
        cls, module: Module, priors: Dict[str, Prior] = None
    ) -> "BayesianConv2d":

        assert isinstance(module, Conv2d)

        # For type checker
        kernel_size = cast(_2_TUPLE_OR_INT, module.kernel_size)
        stride = cast(_2_TUPLE_OR_INT, module.stride)
        padding = cast(_2_TUPLE_OR_INT, module.padding)
        dilation = cast(_2_TUPLE_OR_INT, module.dilation)

        result = cls(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=module.groups,
            bias=module.bias is not None,
            priors=priors,
        )
        result.load_state_dict(module.state_dict(), strict=False)
        return result

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x)


class BayesianNop(BayesianModule):
    def __init__(self) -> None:
        super().__init__()
        self.priors = ModuleDict()

    @classmethod
    def from_freq_module(
        cls, module: Module, priors: Dict[str, Prior]
    ) -> "BayesianNop":
        return cls()

    def forward(self, x: Tensor) -> Tensor:
        return x
