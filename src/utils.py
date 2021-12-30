import importlib
import math
import os
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import cache
from itertools import accumulate, tee
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import pandas as pd
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from optuna import Study
from pytorch_lightning import Callback, Trainer
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch import Size, Tensor
from torch.nn.modules.module import Module

from src.experiments.common import EXPERIMENT_PATH


def import_from(module: str, name: str) -> Any:
    module_ = importlib.import_module(module)
    return getattr(module_, name)


T = TypeVar("T")


def pairwise(iterable: Iterable[T]) -> Iterable[Tuple[T, T]]:
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class ModuleAttributeHelper:
    """Helper for getting/setting module attributes"""

    def __init__(self, module: Module) -> None:
        self.module = module

    def keyed_children(
        self,
    ) -> Union[Iterable[Tuple[int, Module]], Iterable[Tuple[str, Module]]]:

        if isinstance(self.module, nn.Sequential):
            return enumerate(self.module)
        else:
            return self.module.named_children()

    def __getitem__(self, key: Union[int, str]) -> Module:

        if isinstance(self.module, nn.Sequential):
            assert isinstance(key, int)
            return self.module[key]
        else:
            assert isinstance(key, str)
            return getattr(self.module, key)

    def __setitem__(self, key: Union[int, str], value: Module) -> None:

        if isinstance(self.module, nn.Sequential):
            assert isinstance(key, int)
            self.module[key] = value
        else:
            assert isinstance(key, str)
            setattr(self.module, key, value)


class HasForward(Protocol):
    def forward(self, x: Tensor) -> Tensor:
        ...


class SequentialBuilder:

    DUMMY_BATCH_SIZE = 1

    def __init__(self, in_shape: Size) -> None:

        self.in_shape = in_shape
        self.dummy_tensor = torch.randn((self.DUMMY_BATCH_SIZE,) + in_shape)

        self.modules: list[Module] = []

    @property
    def out_shape(self) -> Size:
        return self.dummy_tensor.shape[1:]

    @torch.no_grad()
    def add(self, module: HasForward) -> None:

        self.dummy_tensor = cast(HasForward, module).forward(self.dummy_tensor)
        self.modules.append(cast(Module, module))

    def out_dim(self, dim: int) -> int:
        return self.out_shape[dim]

    def build(self) -> nn.Sequential:
        return nn.Sequential(*self.modules)


class ParameterView:
    def __init__(
        self,
        model: nn.Module,
        parameters: Optional[Iterable[str]] = None,
        buffers: Optional[Iterable[str]] = None,
    ):

        self.model = model

        if parameters is None and buffers is None:
            parameters = [n for n, _ in self.model.named_parameters()]
            buffers = []
        if parameters is None:
            parameters = []
        if buffers is None:
            buffers = []

        self.parameters = set(parameters)
        self.buffers = set(buffers)

        self.param_shapes = {k: x.shape for k, x in self.named_attributes()}

        indices = accumulate(
            self.param_shapes.values(), lambda x, y: x + math.prod(y), initial=0
        )
        self.flat_index_pairs = list(pairwise(indices))

        self.n_params = self.flat_index_pairs[-1][-1]

    def named_attributes(self) -> Iterator[Tuple[str, Tensor]]:

        yield from (
            (n, buffer) for n, buffer in self.model.named_buffers() if n in self.buffers
        )
        yield from (
            (n, parameter)
            for n, parameter in self.model.named_parameters()
            if n in self.parameters
        )
        # yield from ((n, self.model.get_buffer(n)) for n in self.buffers)

    def attributes(self) -> Iterator[Tensor]:
        yield from (
            buffer for n, buffer in self.model.named_buffers() if n in self.buffers
        )
        yield from (
            parameter
            for n, parameter in self.model.named_parameters()
            if n in self.parameters
        )

    def __getitem__(self, key: slice) -> Tensor:

        if type(key) is slice:
            return self._get_slice(key)
        else:
            raise NotImplementedError

    def __setitem__(self, key: slice, value: Tensor) -> None:

        if type(key) is slice:
            self._set_slice(key, value)

    def apply_(self, func: Callable) -> None:
        for attribute in self.attributes():
            func(attribute)

    @property
    def flat_grad(self) -> Tensor:
        with torch.no_grad():
            result = self._flatten(x.grad for x in self.attributes())
        return result

    def _get_slice(self, slice_: slice) -> Tensor:

        if slice_.start is None and slice_.stop is None and slice_.step is None:
            return self._flatten(self.attributes())
        else:
            raise NotImplementedError

    def _set_slice(self, slice_: slice, value: Tensor) -> None:

        if slice_.start is None and slice_.stop is None and slice_.step is None:
            state_dict = self._unflatten(value)
            for name, parameter in self.named_attributes():
                parameter.detach_()
                parameter.copy_(state_dict[name])
        else:
            raise NotImplementedError

    def _flatten(self, tensor_iter: Iterable[Tensor]) -> Tensor:
        return torch.cat([x.flatten() for x in tensor_iter])

    def _unflatten(self, tensor_flat: Tensor) -> Dict[str, Tensor]:
        return {
            k: tensor_flat[a:b].view(shape)
            for (k, shape), (a, b) in zip(
                self.param_shapes.items(), self.flat_index_pairs
            )
        }


def silence_warnings() -> None:
    warnings.filterwarnings(
        "ignore", "`LightningModule.configure_optimizers` returned `None`"
    )
    warnings.filterwarnings(
        "ignore", ".+does not have many workers which may be a bottleneck."
    )
    warnings.filterwarnings("ignore", "The given NumPy array is not writeable,")
    warnings.filterwarnings("ignore", "The number of training samples")
    warnings.filterwarnings(
        "ignore", "The metric 'err/val' is not in the evaluation logs for pruning."
    )


class SilenceWarnings(Callback):
    def on_init_start(self, trainer: Trainer) -> None:
        silence_warnings()


INFERENCE_LABELS = {
    "sgd-map": "SGD (MAP)",
    "sgd-dropout": "SGD (dropout)",
    "sghmc": "SGHMC",
    "sghmc-var-est": "SGHMC (with var. est.)",
    "vi": "VI",
    "vi-exp-weighted": "VI (exp. KL weight)",
}


@dataclass(frozen=True)
class Run:

    dir: Union[str, Path]

    @property
    def _dir(self) -> Path:
        return Path(self.dir)

    @property
    def cfg(self) -> DictConfig:
        return cast(DictConfig, OmegaConf.load(self._dir / ".hydra" / "config.yaml"))

    @cache
    def get_scalar(self, tag: str) -> pd.Series:
        acc = EventAccumulator(str(self._dir.resolve() / "metrics"))
        acc.Reload()

        scalars = acc.Scalars(tag)
        index = [x.step for x in scalars]
        values = [x.value for x in scalars]

        return pd.Series(values, name=tag, index=pd.Index(index, name="step"))

    @property
    def inference_label(self) -> str:
        if "sampler" in self.cfg.inference:
            if "VarianceEstimator" in self.cfg.inference.sampler._target_:
                return INFERENCE_LABELS["sghmc-var-est"]
            else:
                return INFERENCE_LABELS["sghmc"]
        elif "n_particles" in self.cfg.inference:
            if (
                "kl_weighting_scheme" in self.cfg.inference
                and self.cfg.inference.kl_weighting_scheme._target_
                == "src.inference.vi.ExponentialKLWeight"
            ):
                return INFERENCE_LABELS["vi-exp-weighted"]
            else:
                return INFERENCE_LABELS["vi"]
        else:
            if self.cfg.inference.get("use_map", False):
                return INFERENCE_LABELS["sgd-map"]
            else:
                return INFERENCE_LABELS["sgd-dropout"]


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

    def runs(self) -> pd.Series:

        starts = (
            self.study.trials_dataframe()
            .set_index("number")["datetime_start"]
            .rename("trial_start")
        )

        sweep_dir = EXPERIMENT_PATH / "sweeps" / self.study.study_name

        return (
            pd.DataFrame(
                (
                    {
                        "multirun_start": datetime.strptime(
                            f"{x.parents[2].stem}/{x.parents[1].stem}",
                            r"%Y-%m-%d/%H-%M-%S",
                        ),
                        "number": int(x.parents[0].stem),
                        "run": Run(x.parent),
                    }
                    for x in sweep_dir.glob("*/*/*/*.hydra/")
                ),
            )
            .reset_index()
            .join(starts, on="number")
            .loc[lambda x: x.multirun_start <= x.trial_start]
            .sort_values(["number", "multirun_start"])
            .drop_duplicates("number", keep="last")
            .set_index("number")["run"]
        )

    def loss(self) -> pd.DataFrame:
        return pd.concat(
            pd.DataFrame({"err/val": trial.intermediate_values.values()})
            .rename_axis(index="step")
            .assign(trial=trial.number)
            .set_index("trial", append=True)
            for trial in self.study.trials
        )


@contextmanager
def set_directory(path: Path) -> Iterator[None]:
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
