from dataclasses import dataclass
from typing import Dict, Type, Union
from copy import deepcopy
from .modules import BayesianLinear, BayesianModule
from .priors import NormalPrior, ScaleMixturePrior, Prior

from torch.nn import Linear, Module, Sequential
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


_IMPLEMENTED_BAYESIAN_MODULES = [
    {"module": Linear, "bayesian": BayesianLinear, "parameters": ("weight", "bias")}
]

def from_default_prior(prior):
    modules_to_replace = {}
    for implementation in _IMPLEMENTED_BAYESIAN_MODULES:
        module = implementation["module"]
        bayesian = implementation["bayesian"]
        parameters = implementation["parameters"]

        priors = {name: deepcopy(prior) for name in parameters}
        modules_to_replace[module] = BayesianModuleConfig(bayesian, priors)
    return BayesianConversionConfig(modules_to_replace)

_DEFAULT_PRIORS = from_default_prior(NormalPrior())

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
