import torch.nn as nn
from torch.distributions import Normal

import torch
import torch.nn as nn
from src.models.base import Model
from torch.distributions.normal import Normal
from typing import List
from src.utils import HPARAM

class Prior(nn.Module):
    ...

class KnownPrecisionNormalPrior(Prior):

    precision : HPARAM[float]
    mean : HPARAM[float]

    def __init__(self, precision=10., mean=0.):

        super().__init__()

        self.register_buffer("precision", torch.tensor(precision))
        self.register_buffer("mean", torch.tensor(mean))

    def log_prob(self, parameter):
        return(Normal(self.mean, 1. / self.precision.sqrt()).log_prob(parameter))
        

_DEFAULT_PRIORS = {
    nn.Linear : {
        "weight" : {"cls" : KnownPrecisionNormalPrior, "kwargs" : {}},
        "bias" : {"cls" : KnownPrecisionNormalPrior, "kwargs" : {}}, 
    }
}


# Inspired by Pyro https://github.com/pyro-ppl/pyro/blob/dev/pyro/nn/module.py
class _ModuleWithPriorMeta(type):

    _cache = {}

    # Unpickling helper
    class _New:
        def __init__(self, Module : type[nn.Module]):
            self.__class__ = ModuleWithPrior[Module]

    def __getitem__(cls, Module):

        assert isinstance(Module, type)
        assert issubclass(Module, torch.nn.Module)

        if issubclass(Module, ModuleWithPrior):
            return Module

        if Module in _ModuleWithPriorMeta._cache:
            return _ModuleWithPriorMeta._cache[Module]

        class result(Module, ModuleWithPrior): # type: ignore

            def __reduce__(self):
                state = getattr(self, "__getstate__", self.__dict__.copy)()
                return _ModuleWithPriorMeta._New, (Module,), state

            def __init__(self, *args, **kwargs):
                raise NotImplementedError


        result.__name__ = f"{Module.__name__}WithPrior"

        _ModuleWithPriorMeta._cache[Module] = result
        return result


class ModuleWithPrior(nn.Module, metaclass=_ModuleWithPriorMeta):

    priors : nn.ModuleDict

    def prior_log_prob(self : nn.Module):
        return sum(
            prior.log_prob(getattr(self, name)).sum()
            for name, prior in self.priors.items()
        )

def attach_priors_(module : nn.Module, prior_specs = None):

    if prior_specs is None:
        prior_specs = _DEFAULT_PRIORS

    if type(module) not in prior_specs:
        return

    priors = nn.ModuleDict()
    for parameter_name, _ in module.named_parameters():
        spec = _DEFAULT_PRIORS[type(module)][parameter_name]
        priors[parameter_name] = spec["cls"](**spec["kwargs"])

    module.priors = priors
    module.__class__ = ModuleWithPrior[module.__class__]



class _ProbabilisticModelMeta(type):

    _cache = {}

    # Unpickling helper
    class _New:
        def __init__(self, Module : type[Model]):
            self.__class__ = ProbabilisticModel[Module]

    def __getitem__(cls, Module):

        assert isinstance(Module, type)
        assert issubclass(Module, Model)

        if issubclass(Module, ProbabilisticModel):
            return Module

        if Module in _ProbabilisticModelMeta._cache:
            return _ProbabilisticModelMeta._cache[Module]

        class result(Module, ProbabilisticModel): # type: ignore

            def __reduce__(self):
                state = getattr(self, "__getstate__", self.__dict__.copy)()
                return _ProbabilisticModelMeta._New, (Module,), state

            def __init__(self, *args, **kwargs):
                raise NotImplementedError

        result.__name__ = f"Probabilistic{Module.__name__}"

        _ProbabilisticModelMeta._cache[Module] = result
        return result

class ProbabilisticModel(Model, metaclass = _ProbabilisticModelMeta): 

    submodules_with_prior : List[ModuleWithPrior]

    def log_prior(self):
        """Returns p(theta)"""
        return sum(m.prior_log_prob() for m in self.submodules_with_prior)

    def log_likelihood(self, x, y):
        """Returns log p(y | x, theta)"""
        return self.observation_model(x).log_prob(y)

def to_probabilistic_model_(model : Model, prior_specs=None):

    assert(isinstance(model, Model))

    submodules_with_prior = []
    for module in model.modules():
        attach_priors_(module, prior_specs)
        if isinstance(module, ModuleWithPrior):
            submodules_with_prior.append(module)

    model.submodules_with_prior = submodules_with_prior
    model.__class__ = ProbabilisticModel[model.__class__]
