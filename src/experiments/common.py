from inspect import signature
import os
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Optional, Union
from typing_extensions import runtime

import torch
from omegaconf import Container, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.saving import save_hparams_to_yaml
from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
from pytorch_lightning.loggers.csv_logs import ExperimentWriter
from pytorch_lightning.utilities import (
    _OMEGACONF_AVAILABLE,
    rank_zero_only,
    rank_zero_warn,
)
from pytorch_lightning.utilities.cloud_io import get_filesystem
from pytorch_lightning.utilities.distributed import rank_zero_only

# from src.utils import Component, HPARAM
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams
import logging
import re
from hydra.utils import get_original_cwd, to_absolute_path


ROOT_DIR = Path(__file__).parents[2]


class FlatTensorBoardLogger(LightningLoggerBase):

    NAME_HPARAMS_FILE = "hparams.yaml"
    LOGGER_JOIN_CHAR = "-"

    def __init__(
        self,
        save_dir: str,
        log_graph: bool = False,
        default_hp_metric: bool = True,
        prefix: str = "",
        **kwargs,
    ):
        super().__init__()
        self._save_dir = save_dir
        self._log_graph = log_graph
        self._default_hp_metric = default_hp_metric
        self._prefix = prefix
        self._fs = get_filesystem(save_dir)

        self._experiment = None
        self.hparams = {}
        self._kwargs = kwargs

    @property
    def save_dir(self) -> Optional[str]:
        return self._save_dir

    @property
    @rank_zero_experiment
    def experiment(self):
        r"""
        Actual tensorboard object. To use TensorBoard features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.

        Example::

            self.logger.experiment.some_tensorboard_function()

        """
        if self._experiment is not None:
            return self._experiment

        assert rank_zero_only.rank == 0, "tried to init log dirs in non global_rank=0"

        self._experiment = SummaryWriter(log_dir=self.save_dir, **self._kwargs)
        return self._experiment

    @rank_zero_only
    def log_hyperparams(
        self,
        params: Union[Dict[str, Any], Namespace],
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record hyperparameters. TensorBoard logs with and without saved hyperparameters
        are incompatible, the hyperparameters are then not displayed in the TensorBoard.
        Please delete or move the previously saved logs to display the new ones with hyperparameters.

        Args:
            params: a dictionary-like container with the hyperparameters
            metrics: Dictionary with metric names as keys and measured quantities as values
        """

        params = self._convert_params(params)

        # store params to output
        if _OMEGACONF_AVAILABLE and isinstance(params, Container):
            self.hparams = OmegaConf.merge(self.hparams, params)
        else:
            self.hparams.update(params)

        # format params into the suitable for tensorboard
        params = self._flatten_dict(params)
        params = self._sanitize_params(params)

        if metrics is None:
            if self._default_hp_metric:
                metrics = {"hp_metric": -1}
        elif not isinstance(metrics, dict):
            metrics = {"hp_metric": metrics}

        if metrics:
            self.log_metrics(metrics, 0)
            exp, ssi, sei = hparams(params, metrics)
            writer = self.experiment._get_file_writer()
            writer.add_summary(exp)
            writer.add_summary(ssi)
            writer.add_summary(sei)

    @rank_zero_only
    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"

        metrics = self._add_prefix(metrics)

        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            if isinstance(v, dict):
                self.experiment.add_scalars(k, v, step)
            else:
                try:
                    self.experiment.add_scalar(k, v, step)
                # todo: specify the possible exception
                except Exception as ex:
                    m = f"\n you tried to log {v} which is not currently supported. Try a dict or a scalar/tensor."
                    raise ValueError(m) from ex

    @rank_zero_only
    def log_graph(self, model: "pl.LightningModule", input_array=None):
        if self._log_graph:
            if input_array is None:
                input_array = model.example_input_array

            if input_array is not None:
                input_array = model._apply_batch_transfer_handler(input_array)
                self.experiment.add_graph(model, input_array)
            else:
                rank_zero_warn(
                    "Could not log computational graph since the"
                    " `model.example_input_array` attribute is not set"
                    " or `input_array` was not given",
                    UserWarning,
                )

    @rank_zero_only
    def save(self) -> None:
        super().save()
        dir_path = self.save_dir

        # prepare the file path
        hparams_file = os.path.join(dir_path, self.NAME_HPARAMS_FILE)

        # save the metatags file if it doesn't exist and the log directory exists
        if self._fs.isdir(dir_path) and not self._fs.isfile(hparams_file):
            save_hparams_to_yaml(hparams_file, self.hparams)

    @rank_zero_only
    def finalize(self, status: str) -> None:
        self.experiment.flush()
        self.experiment.close()
        self.save()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_experiment"] = None
        return state

    @property
    def name(self):
        pass

    @property
    def version(self):
        pass


from enum import Enum, auto

class PlotType(Enum):
    SINGLE_RUN = auto()
    MULTI_RUN = auto()

def plot(multirun=False):

    if multirun:
        plot_type = PlotType.MULTI_RUN
    else:
        plot_type = PlotType.SINGLE_RUN

    def decorator(func):
        func.__isplot = True
        func.__plot_type = plot_type
        return func

    return decorator


def result(func):
    func.__isresult = True
    return func




import datetime
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf


def flatten_config(config):
    def iter_flat_config(config, prefixes=None):
        if prefixes is None:
            prefixes = []

        if OmegaConf.is_list(config):
            k_v_iter = ((str(i), x) for i, x in enumerate(config))
        elif OmegaConf.is_dict(config):
            k_v_iter = config.items()

        for k, v in k_v_iter:
            if OmegaConf.is_config(v):
                yield from iter_flat_config(v, prefixes=prefixes + [k])
            elif k == "_target_":
                yield "/".join(prefixes), v
            else:
                yield "/".join(prefixes + [k]), v

    return dict(sorted(list(iter_flat_config(config))))


from dataclasses import dataclass, asdict

EXPERIMENT_PATH = Path(__file__).parents[2] / "experiment_results"

import datetime
from dataclasses import dataclass, field

from omegaconf import DictConfig

from typing import List


from contextlib import contextmanager
from pathlib import Path

import os


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


from functools import cache

log = logging.getLogger(__name__)


@dataclass(order=True, frozen=True)
class Run:

    experiment_name: str
    time: datetime.datetime
    config: DictConfig = field(repr=False, compare=False)
    overrides: DictConfig = field(repr=False, compare=False)
    path: Path = field(repr=False, compare=False)
    run_index: Optional[int]

    @property
    def experiment(self):
        return Experiment(self.experiment_name)

    @property
    def short_name(self):
        name = self.experiment_name
        if self.run_index is not None:
            name += f"[{self.run_index}]"
        return name

    @cache
    def load_result(self, result_func_name=None):

        log.info(f"Loading <{result_func_name}> for run {self.short_name}")
        result_func = self.experiment.get_result_funcs()[result_func_name]
        with set_directory(self.path):
            results = result_func()
        return results

    def call_plot_func(self, plot_func):

        if getattr(plot_func, ("__plot_type")) == PlotType.SINGLE_RUN:

            kwargs = {}
            for arg_name in signature(plot_func).parameters:
                if arg_name == "_run_":
                    kwargs["_run_"] = self
                else:
                    kwargs[arg_name] = self.load_result(arg_name)
            log.info(f"Plotting single run plot: <{plot_func.__name__}>")

            plot_func(**kwargs)
        else:
            data = asdict(self)
            data["run_index"] = 0
            multirun = MultiRun(
                self.experiment_name, self.time, [Run(**data)], self.path
            )
            try:
                multirun.call_plot_func(plot_func)
            except:
                log.info(
                    f"Plotting single run as multi run failed. Skipping <{plot_func.__name__}>"
                )

    def get_override_value(self, value):

        match = next(
            y
            for y in (re.match(f"{value}=(.+)", x) for x in self.overrides)
            if y is not None
        )
        return match.group(1)


@dataclass(order=True)
class MultiRun:

    experiment_name: str
    time: datetime.datetime
    runs: List[Run] = field(compare=False)
    path: Path = field(repr=False, compare=False)

    @property
    def experiment(self):
        return Experiment(self.experiment_name)

    def collect_results(self, result_func_name):
        return {i: x.load_result(result_func_name) for i, x in enumerate(self.runs)}

    def call_plot_func(self, plot_func):

        if getattr(plot_func, ("__plot_type")) == PlotType.MULTI_RUN:
            kwargs = {}
            for arg_name in signature(plot_func).parameters:
                if arg_name == "_run_":
                    kwargs["_run_"] = dict(enumerate(self.runs))
                else:
                    kwargs[arg_name] = self.collect_results(arg_name)

            log.info(f'Plotting multi run plot: "{plot_func.__name__}"')
            plot_func(**kwargs)

        elif getattr(plot_func, ("__plot_type")) == PlotType.SINGLE_RUN:
            for run in self.runs:
                out_dir = (Path(".") / f"{run.run_index}").resolve()
                out_dir.mkdir(exist_ok=True)
                with (set_directory(out_dir)):
                    run.call_plot_func(plot_func)


def get_run_from_path(path):

    path = Path(to_absolute_path(path))

    name, day, time, *run_index = str(path.relative_to(EXPERIMENT_PATH)).split("/")
    time = datetime.datetime.strptime(f"{day}/{time}", r"%Y-%m-%d/%H-%M-%S")

    if (path / ".hydra").exists():
        config = OmegaConf.load(path / ".hydra" / "config.yaml")
        overrides = OmegaConf.load(path / ".hydra" / "overrides.yaml")
        run_index = int(run_index[0]) if len(run_index) > 0 else None
        return Run(name, time, config, overrides, path, run_index)
    else:
        runs = [
            get_run_from_path(x)
            for x in path.iterdir()
            if x.stem.isdigit() and x.is_dir()
        ]
        if len(runs) > 0:
            return MultiRun(name, time, runs, path)


from importlib import import_module
from inspect import getmembers


@dataclass
class Experiment:

    name: str

    @property
    def path(self):
        return EXPERIMENT_PATH / self.name

    def run_dirs(self):
        return sorted([x for x in self.path.glob(r"*/*") if x.is_dir()])

    def runs(self):
        return [get_run_from_path(dir_) for dir_ in self.run_dirs()]

    def members(self):
        experiment_module = import_module(f"src.experiments.{self.name}")
        return getmembers(experiment_module)

    def get_plot_funcs(self):
        return {name: x for name, x in self.members() if getattr(x, "__isplot", False)}

    def get_result_funcs(self):
        return {
            name: x for name, x in self.members() if getattr(x, "__isresult", False)
        }

    