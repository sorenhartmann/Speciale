import importlib
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import accumulate, tee
from pathlib import Path
from typing import Union

import pandas as pd
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from optuna import Study
from tensorboard.backend.event_processing.event_accumulator import \
    EventAccumulator


def import_from(module, name):
    module = importlib.import_module(module)
    return getattr(module, name)

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

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
            return getattr(self.module, key)

    def __setitem__(self, key, value):

        if isinstance(self.module, nn.Sequential):
            self.module[key] = value
        else:
            setattr(self.module, key, value)

class SequentialBuilder:

    DUMMY_BATCH_SIZE = 1

    def __init__(self, in_shape):

        self.in_shape = in_shape
        self.dummy_tensor = torch.randn((self.DUMMY_BATCH_SIZE,) + in_shape)

        self.modules = []

    @property
    def out_shape(self):
        return self.dummy_tensor.shape[1:]

    @torch.no_grad()
    def add(self, module):

        self.dummy_tensor = module(self.dummy_tensor)
        self.modules.append(module)

    def out_dim(self, dim):
        return self.out_shape[dim]

    def build(self):
        return nn.Sequential(*self.modules)


class ParameterView:

    def __init__(self, model: nn.Module, parameters=None, buffers=None):

        self.model = model

        if parameters is None and buffers is None:
            parameters = [n for n, _ in self.model.named_parameters()]
            buffers = []
        elif parameters is None:
            parameters = []
        elif buffers is None:
            buffers = []

        self.parameters = set(parameters)
        self.buffers = set(buffers)

        self.param_shapes = {k: x.shape for k, x in self.named_attributes()}

        indices = accumulate(
            self.param_shapes.values(), lambda x, y: x + math.prod(y), initial=0
        )
        self.flat_index_pairs = list(pairwise(indices))

        self.n_params = self.flat_index_pairs[-1][-1]

    def named_attributes(self):

        yield from (
            (n, buffer) for n, buffer in self.model.named_buffers() if n in self.buffers
        )
        yield from (
            (n, parameter)
            for n, parameter in self.model.named_parameters()
            if n in self.parameters
        )
        # yield from ((n, self.model.get_buffer(n)) for n in self.buffers)

    def attributes(self):
        yield from (
            buffer for n, buffer in self.model.named_buffers() if n in self.buffers
        )
        yield from (
            parameter
            for n, parameter in self.model.named_parameters()
            if n in self.parameters
        )

    def __getitem__(self, key):

        if type(key) is slice:
            return self._get_slice(key)
        else:
            raise NotImplementedError

    def __setitem__(self, key, value):

        if type(key) is slice:
            self._set_slice(key, value)

    def apply_(self, func):
        for attribute in self.attributes():
            func(attribute)

    @property
    @torch.no_grad()
    def flat_grad(self):
        return self._flatten(x.grad for x in self.attributes())

    def _get_slice(self, slice_):

        if slice_.start is None and slice_.stop is None and slice_.step is None:
            return self._flatten(self.attributes())
        else:
            raise NotImplementedError

    def _set_slice(self, slice_, value):

        if slice_.start is None and slice_.stop is None and slice_.step is None:
            state_dict = self._unflatten(value)
            for name, parameter in self.named_attributes():

                #     if parameter.requires_grad:
                #         parameter.copy_(state_dict[name])
                # else:
                parameter.detach_()
                parameter.copy_(state_dict[name])
        else:
            raise NotImplementedError

    def _flatten(self, tensor_iter):
        return torch.cat([x.flatten() for x in tensor_iter])

    def _unflatten(self, value):
        return {
            k: value[a:b].view(shape)
            for (k, shape), (a, b) in zip(
                self.param_shapes.items(), self.flat_index_pairs
            )
        }


import warnings


def silence_warnings():
    warnings.filterwarnings("ignore", "`LightningModule.configure_optimizers` returned `None`")
    warnings.filterwarnings("ignore", ".+does not have many workers which may be a bottleneck.")
    warnings.filterwarnings("ignore", "The given NumPy array is not writeable,")
    warnings.filterwarnings("ignore", "The number of training samples")
    warnings.filterwarnings("ignore", "The metric 'err/val' is not in the evaluation logs for pruning.")

from pytorch_lightning import Callback


class SilenceWarnings(Callback):

    def on_init_start(self, trainer):
        silence_warnings()

from functools import cache

@dataclass(frozen=True)
class Run:
    dir: Union[str, Path]

    @property
    def cfg(self):
        return OmegaConf.load(self.dir / ".hydra" / "config.yaml")

    @cache
    def get_scalar(self, tag):
        acc = EventAccumulator(str(self.dir.resolve() / "metrics"))
        acc.Reload()

        scalars = acc.Scalars(tag)
        index = [x.step for x in scalars]
        values = [x.value for x in scalars]

        return pd.Series(values, name=tag, index=pd.Index(index, name="step"))

    @property
    def inference_label(self):
        if "sampler" in self.cfg.inference:
            if "VarianceEstimator" in self.cfg.inference.sampler._target_:
                return "SGHMC (with var. est.)"
            else:
                return "SGHMC"
        elif "n_particles" in self.cfg.inference:
            return "VI"
        else:
            if self.cfg.model.dropout is None or self.cfg.model.dropout == 0:
                return "SGD (MAP)"            
            else:
                return "SGD (dropout)"


@dataclass
class Sweep:
    study: Study

    def summary(self) -> pd.DataFrame:

        return pd.DataFrame.from_records(
            (
                {
                    "trial": trial.number,
                    "datetime_start": trial.datetime_start,
                    "err/val": trial.value,
                    **trial.params,
                }
                for trial in self.study.trials
            ),
            index="trial",
        ).sort_values("err/val")

    def loss(self) -> pd.DataFrame:
        return pd.concat(
            pd.DataFrame({"err/val": trial.intermediate_values.values()})
            .rename_axis(index="step")
            .assign(trial=trial.number)
            .set_index("trial", append=True)
            for trial in self.study.trials
        )


@contextmanager
def set_directory(path: Path):
    """Sets the cwd within the context

    Args:
        path (Path): The path to the cwd

    Yields:
        None
    """

    origin = Path().absolute()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(origin)
