from copy import deepcopy
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions.normal import Normal

from src.models.base import Model


class Prior(nn.Module):
    ...


class KnownPrecisionNormalPrior(Prior):
    def __init__(self, precision=1.0, mean=0.0):

        super().__init__()

        self.register_buffer("precision", torch.tensor(precision))
        self.register_buffer("mean", torch.tensor(mean))

    def log_prob(self, parameter):
        return Normal(self.mean, 1.0 / self.precision.sqrt()).log_prob(parameter)

from torch.distributions import Bernoulli, Categorical, MixtureSameFamily


class NormalMixturePrior(Prior):
    def __init__(
        self, mean=[0.0, 0.0], log_sigma=[-1, -7], mixture_ratio=0.5
    ):

        super().__init__()

        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("log_sigma", torch.tensor(log_sigma))
        mixture_logits = torch.tensor([mixture_ratio, 1-mixture_ratio])
        self.register_buffer("mixture_logits", mixture_logits)

    def log_prob(self, parameter):

        dist = MixtureSameFamily(
            Categorical(self.mixture_logits),
            Normal(self.mean, self.log_sigma.exp()),
            )

        return dist.log_prob(parameter)

from dataclasses import dataclass, field
from typing import Dict, Type


@dataclass
class PriorSpec:

    default_prior: Prior
    modules_to_replace: List[Type[nn.Module]] = field(
        default_factory=lambda: [nn.Linear, nn.Conv2d]
    )
    overrides: Dict[Tuple[Type[nn.Module], str], Prior] = field(default_factory=dict)

    def __getitem__(self, key):
        module_class, str = key
        prior = self.overrides.get(key, self.default_prior)
        return deepcopy(prior)

    def __contains__(self, key):
        return key in self.modules_to_replace


_DEFAULT_PRIORS = PriorSpec(KnownPrecisionNormalPrior())


class ModuleWithPrior(nn.Module):

    priors: nn.ModuleDict

    def __init__(self, module, priors: nn.ModuleDict):

        super().__init__()
        self.module = module
        self.priors = priors

    def forward(self, x):
        return self.module.forward(x)

    def prior_log_prob(self: nn.Module):
        return sum(
            prior.log_prob(getattr(self.module, name)).sum()
            for name, prior in self.priors.items()
        )


def with_priors(module: nn.Module, prior_specs):

    module_class = type(module)

    if module_class not in prior_specs:
        return

    priors = nn.ModuleDict()
    for parameter_name, _ in module.named_parameters():
        priors[parameter_name] = prior_specs[module_class, parameter_name]

    return ModuleWithPrior(module, priors)


class ProbabilisticModel(Model):

    model: Model
    submodules_with_prior: List[ModuleWithPrior]

    def __init__(self, model, submodules_with_prior):

        super().__init__()
        self.model = model
        self.submodules_with_prior = submodules_with_prior

    def observation_model_gvn_output(self, output):
        return self.model.observation_model_gvn_output(output)

    def observation_model(self, input):
        return self.model.observation_model(input)

    def loss(self, output, target):
        return self.model.loss(output, target)

    def get_metrics(self):
        return self.model.get_metrics()

    def predict(self, x):
        return self.model.predict(x)

    def forward(self, x):
        return self.model.forward(x)

    def log_prior(self):
        """Returns p(theta)"""
        return sum(m.prior_log_prob() for m in self.submodules_with_prior)

    def log_likelihood(self, x, y):
        """Returns log p(y | x, theta)"""
        return self.observation_model(x).log_prob(y)


class ModuleAttributeHelper:
    """Helper for getting/setting module attributes"""

    def __init__(self, module):
        self.module = module

    def keyed_children(self):

        if isinstance(self.module, nn.Sequential):
            return enumerate(self.module)
        else:
            return self.module.named_children()

    def __getitem__(self, key):

        if isinstance(self.module, nn.Sequential):
            return self.module[key]
        else:
            return getattr(self.module)

    def __setitem__(self, key, value):

        if isinstance(self.module, nn.Sequential):
            self.module[key] = value
        else:
            setattr(self.module, key, value)


def as_probabilistic_model(model: Model, prior_spec=None):

    if prior_spec is None:
        prior_spec = _DEFAULT_PRIORS

    assert isinstance(model, Model)
    model = deepcopy(model)

    submodules_with_prior = []

    def replace_submodules_(module: nn.Module):

        helper = ModuleAttributeHelper(module)
        for key, child in helper.keyed_children():
            if type(child) in prior_spec:
                new_module = with_priors(child, prior_spec)
                helper[key] = new_module
                submodules_with_prior.append(new_module)
            else:
                replace_submodules_(child)

    replace_submodules_(model)

    return ProbabilisticModel(model, submodules_with_prior)
