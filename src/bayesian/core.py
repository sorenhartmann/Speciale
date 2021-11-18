from dataclasses import dataclass
from typing import Dict, Type, Union
from copy import deepcopy
from .modules import (
    BayesianConv2d,
    BayesianLinear,
    BayesianModule,
    BayesianNop,
    BayesianScaleShift,
)
from .priors import NormalPrior, ScaleMixturePrior, Prior

from torch.nn import Linear, Module, Sequential, BatchNorm2d, Conv2d
from src.models.base import Model
from src.utils import ModuleAttributeHelper


@dataclass
class BayesianModuleConfig:
    module: Type[BayesianModule]
    priors: Dict[str, Prior]

    def get_prior(self):
        return deepcopy(self.priors)

@dataclass
class BayesianConversionConfig:
    modules_to_replace: Dict[Type[Module], BayesianModuleConfig]


def from_flat_conversion_config(flat_config):
    return BayesianConversionConfig({x["module"] : x["config"] for x in flat_config})

_flat_default = [
    {
        "module": Linear,
        "config": BayesianModuleConfig(
            module=BayesianLinear,
            priors={
                "weight": NormalPrior(),
                "bias": NormalPrior(),
            },
        ),
    },
    {
        "module": Conv2d,
        "config": BayesianModuleConfig(
            module=BayesianConv2d,
            priors={
                "weight": NormalPrior(),
                "bias": NormalPrior(),
            },
        ),
    },
    {
        "module": BatchNorm2d,
        "config": BayesianModuleConfig(
            module=BayesianNop,
            priors={},
        ),
    },
]

_DEFAULT_PRIORS = from_flat_conversion_config(_flat_default)

def to_bayesian_model(model: Model, conversion_config: BayesianConversionConfig = None):
    """Replaces submodules with bayesian modules"""
    assert isinstance(model, Model)
    model = deepcopy(model)

    if conversion_config is None:
        conversion_config = _DEFAULT_PRIORS

    def replace_submodules_(module: Module):

        helper = ModuleAttributeHelper(module)
        for key, child in helper.keyed_children():
            bayesian_config = conversion_config.modules_to_replace.get(type(child))
            if bayesian_config is not None:
                priors = bayesian_config.get_prior()
                new_module = bayesian_config.module.from_freq_module(child, priors)
                helper[key] = new_module
            else:
                replace_submodules_(child)

    replace_submodules_(model)

    return model


def iter_bayesian_modules(module):
    for child in module.children():
        if isinstance(child, BayesianModule):
            yield child
        else:
            yield from iter_bayesian_modules(child)


def log_prior(model: Model):
    """Returns p(theta)"""
    return sum(x.log_prior() for x in iter_bayesian_modules(model))


def log_likelihood(model: Model, x, y):
    """Returns log p(y | x, theta)"""
    return model.observation_model(x).log_prob(y)
