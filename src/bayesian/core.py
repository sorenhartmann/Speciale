from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Type

import torch
from torch import Tensor
from torch.nn import BatchNorm2d, Conv2d, Linear, Module

from src.models.base import Model
from src.utils import ModuleAttributeHelper

from .modules import BayesianConv2d, BayesianLinear, BayesianModule, BayesianNop
from .priors import NormalPrior, Prior


@dataclass
class BayesianModuleConfig:
    module: Type[BayesianModule]
    priors: Dict[str, Prior]

    def get_prior(self) -> Dict[str, Prior]:
        return deepcopy(self.priors)


@dataclass
class BayesianConversionConfig:
    modules_to_replace: Dict[Type[Module], BayesianModuleConfig]


_DEFAULT_PRIORS = BayesianConversionConfig(
    {
        Linear: BayesianModuleConfig(
            module=BayesianLinear,
            priors={
                "weight": NormalPrior(),
                "bias": NormalPrior(),
            },
        ),
        Conv2d: BayesianModuleConfig(
            module=BayesianConv2d,
            priors={
                "weight": NormalPrior(),
                "bias": NormalPrior(),
            },
        ),
        BatchNorm2d: BayesianModuleConfig(
            module=BayesianNop,
            priors={},
        ),
    }
)


def to_bayesian_model(
    model: Model,
    conversion_config: Optional[BayesianConversionConfig] = None,
) -> Model:
    """Replaces submodules with bayesian modules"""

    assert isinstance(model, Model)
    model = deepcopy(model)

    if conversion_config is None:
        conversion_config = _DEFAULT_PRIORS

    def replace_submodules_(
        module: Module,
        conversion_config: BayesianConversionConfig,
    ) -> None:

        helper = ModuleAttributeHelper(module)
        for key, child in helper.keyed_children():
            bayesian_config = conversion_config.modules_to_replace.get(type(child))
            if bayesian_config is not None:
                priors = bayesian_config.get_prior()
                new_module = bayesian_config.module.from_freq_module(child, priors)
                helper[key] = new_module
            else:
                replace_submodules_(child, conversion_config)

    replace_submodules_(model, conversion_config)

    return model


def iter_bayesian_modules(module: Module) -> Iterator[BayesianModule]:
    for child in module.children():
        if isinstance(child, BayesianModule):
            yield child
        else:
            yield from iter_bayesian_modules(child)


def log_prior(model: Model) -> Tensor:
    """Returns p(theta)"""
    return sum((x.log_prior() for x in iter_bayesian_modules(model)), torch.tensor(0))


def log_likelihood(model: Model, x: Tensor, y: Tensor) -> Tensor:
    """Returns log p(y | x, theta)"""
    return model.observation_model(x).log_prob(y)
