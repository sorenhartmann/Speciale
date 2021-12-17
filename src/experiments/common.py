import logging
import os
import re
from argparse import Namespace
from inspect import signature
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from hydra.utils import get_original_cwd, to_absolute_path
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

from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams
from typing_extensions import runtime

from inspect import signature


from dataclasses import asdict
from functools import wraps


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
            k_v_iter = config.items

        for k, v in k_v_iter:
            if OmegaConf.is_config(v):
                yield from iter_flat_config(v, prefixes=prefixes + [k])
            elif k == "_target_":
                yield "/".join(prefixes), v
            else:
                yield "/".join(prefixes + [k]), v

    return dict(sorted(list(iter_flat_config(config))))


from dataclasses import asdict, dataclass

EXPERIMENT_PATH = Path(__file__).parents[2] / "experiment_results"

import datetime
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from omegaconf import DictConfig





from functools import cache

log = logging.getLogger(__name__)


@dataclass(order=True, frozen=True)
class Run:

    experiment_name: str
    date: datetime.date
    id: str
    config: DictConfig = field(repr=False, compare=False)
    overrides: DictConfig = field(repr=False, compare=False)
    path: Path = field(repr=False, compare=False)

    @property
    def experiment(self):
        return Experiment(self.experiment_name)

    @property
    def short_name(self):
        name = self.experiment_name
        if self.run_index is not None:
            name += f"[{self.run_index}]"
        return name

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
    date: datetime.date
    id: str
    runs: List[Run] = field(compare=False)
    path: Path = field(repr=False, compare=False)

    @property
    def experiment(self):
        return Experiment(self.experiment_name)


def get_run_from_path(path, recurse=True):

    path = Path(to_absolute_path(path))

    name, date, *id_ = str(path.relative_to(EXPERIMENT_PATH)).split("/")
    date = datetime.date.fromisoformat(date)
    id_ = "/".join(id_)

    if (path / ".hydra").exists():
        config = OmegaConf.load(path / ".hydra" / "config.yaml")
        overrides = OmegaConf.load(path / ".hydra" / "overrides.yaml")
        return Run(name, date, id_, config, overrides, path)
    elif recurse:
        runs = []
        for x in path.iterdir():
            if not x.is_dir() or x.stem.startswith("_"):
                continue
            run = get_run_from_path(x, recurse=False)
            if run is not None:
                runs.append(run)

        if len(runs) > 0:
            runs = sorted(
                runs,
                key=lambda x: f"{int(x.path.stem):03}"
                if x.path.stem.isnumeric()
                else x.path.stem,
            )
            return MultiRun(name, date, id_, runs, path)
    else:
        return None


def cast_run(func):

    sig = signature(func)

    multirun_args = [
        (a, k.annotation) for a, k in sig.parameters.items() if k.annotation is MultiRun
    ]
    singlerun_args = [
        (a, k.annotation) for a, k in sig.parameters.items() if k.annotation is Run
    ]
    ((arg_name, arg_type),) = multirun_args + singlerun_args

    @wraps(func)
    def wrapper(run, *args, **kwargs):

        if isinstance(run, arg_type):
            func(run, *args, **kwargs)
        elif arg_type is MultiRun:
            data = asdict(run)
            data["run_index"] = 0
            multirun = MultiRun(run.experiment_name, run.time, [Run(**data)], run.path)
            func(multirun, *args, **kwargs)
        elif arg_type is Run:
            for run_ in run.runs:
                out_dir = (Path(".") / f"{run_.run_index}").resolve()
                out_dir.mkdir(exist_ok=True)
                with (set_directory(out_dir)):
                    func(run_, *args, **kwargs)

    return wrapper


from importlib import import_module
from inspect import getmembers
import os


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

    def latest_run(self):
        run_dirs = self.run_dirs()
        latest_dir = run_dirs[-1]
        return get_run_from_path(latest_dir.absolute())

    def members(self):
        experiment_module = import_module(f"src.experiments.{self.name}")
        return getmembers(experiment_module)

    def get_plot_funcs(self):
        return {name: x for name, x in self.members() if getattr(x, "__isplot", False)}

    def get_result_funcs(self):
        return {
            name: x for name, x in self.members() if getattr(x, "__isresult", False)
        }

