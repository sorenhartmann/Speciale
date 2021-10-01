from abc import abstractmethod, ABC
from contextlib import contextmanager
import contextlib
import datetime
import inspect
import pickle
import os
from src.inference import BayesianClassifier
from src.modules import ProbabilisticModel
import torch
import yaml
from collections.abc import Callable
from pytorch_lightning.loggers.csv_logs import ExperimentWriter
from src.samplers import (
    Hamiltonian,
    MetropolisHastings,
    StochasticGradientHamiltonian,
    HamiltonianNoMH,
)
import argparse
from typing import Any, Dict, Optional, Type, Union
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities.cloud_io import get_filesystem
samplers = {
    sampler_cls.tag: sampler_cls
    for sampler_cls in [
        MetropolisHastings,
        Hamiltonian,
        StochasticGradientHamiltonian,
        HamiltonianNoMH,
    ]
}
import re
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import Namespace
import pandas as pd
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.utilities import _OMEGACONF_AVAILABLE, rank_zero_only, rank_zero_warn
if _OMEGACONF_AVAILABLE:
    from omegaconf import Container, OmegaConf  # type: ignore
from pytorch_lightning.core.saving import save_hparams_to_yaml
from torch.utils.tensorboard.summary import hparams
import logging

log = logging.getLogger(__name__)


ROOT_DIR = Path(__file__).parents[2]


class GetSampler(argparse.Action):

    default = Hamiltonian
    samplers = samplers

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, self.samplers[values])


def get_args(model_cls: Type[ProbabilisticModel], inference_cls: Type[BayesianClassifier]):

    parser = argparse.ArgumentParser()

    # Batch size (for batched inference only)
    parser.add_argument("--batch_size", type=int, default=8)

    # Sampler and sampler args
    parser.add_argument(
        "--sampler",
        action=GetSampler,
        default=GetSampler.default,
        choices=GetSampler.samplers.keys(),
    )
    known_args, _ = parser.parse_known_args()
    parser = known_args.sampler.add_argparse_args(parser)

    # Model specific args
    parser = model_cls.add_argparse_args(parser)

    # Inference specific args
    parser = inference_cls.add_argparse_args(parser)

    # Training specific args
    parser = Trainer.add_argparse_args(parser)

    # Parse and return
    args = parser.parse_args()
    return args

_idx_match = re.compile("run_(\\d+)").match

@contextlib.contextmanager
def working_directory(path):
    """Changes working directory and returns to previous on exit."""
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)

EXP_DIR = ROOT_DIR / "experiment_results" 

class Run:

    def __init__(self, run_dir=None, experiment_name=None, run_id=None):
        
        if run_dir is not None:
            self.dir = Path(run_dir)
        elif experiment_name is not None and run_id is not None:
            self.dir = EXP_DIR / experiment_name / f"run_{run_id}"
        else:
            raise FileNotFoundError

        assert self.dir.exists()
        
    @property
    def metrics(self) -> pd.DataFrame:
        return (
            pd.read_csv(self.dir / "metrics.csv")
            .groupby(["epoch", "step"])
            .agg(lambda x: x.loc[x.first_valid_index()])
        )

    @property
    def hparams(self) -> Dict[str, Any]:
        with open(self.dir / "hparams.yaml") as f:
            hparams = yaml.load(f)
        return hparams

    @property
    def result(self) -> Any:
        with open(self.dir / "results.pkl", "rb") as f:
            result = pickle.load(f)
        return result

class ExperimentHandler:

    HPARAM_FILE = "hparams.yaml"
    RESULTS_FILE = "results.pkl"

    def __init__(self, experiment : Callable[..., Optional[pd.DataFrame]]):

        self.experiment = experiment

        self.name = Path(inspect.getfile(experiment)).stem
        self.dir = EXP_DIR / self.name
        self.dir.mkdir(exist_ok=True)
        self.conf = None

    def run_dirs(self):
        return [dir_ for dir_ in self.dir.iterdir() if _idx_match(dir_.name)]

    def run(self, **experiment_kwargs):

        run_dirs = self.run_dirs()

        if len(run_dirs) == 0:
            run_id = 1
        else:
            run_id = max(int(_idx_match(dir_.name).group(1)) for dir_ in run_dirs) + 1

        run_dir = self.dir / f"run_{run_id}"
        run_dir.mkdir()

        with working_directory(run_dir):
            results = self.experiment(**experiment_kwargs)

        if results is not None:
            with open(run_dir / self.RESULTS_FILE, "wb") as f:
                pickle.dump(results, f)

        if not Path(run_dir / self.HPARAM_FILE).exists():
            signature = inspect.signature(self.experiment)
            hparams = {name : p.default for name, p in signature.parameters.items()}
            hparams.update(experiment_kwargs)
            with open(run_dir / self.HPARAM_FILE, "w") as f:
                yaml.dump(hparams, f)

    def latest_run(self):

        run_dirs = sorted(self.run_dirs())
        latest_run_dir = run_dirs[-1]
        return Run(latest_run_dir)

    def runs(self):
        pass


class FlatCSVLogger(LightningLoggerBase):

    def __init__(
        self,
        save_dir: str,
    ):
        super().__init__()
        self._save_dir = save_dir
        self._experiment = None
        self._prefix = ""

    @property
    def save_dir(self) -> Optional[str]:
        return self._save_dir

    @property
    @rank_zero_experiment
    def experiment(self) -> ExperimentWriter:
        r"""

        Actual ExperimentWriter object. To use ExperimentWriter features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.

        Example::

            self.logger.experiment.some_experiment_writer_function()

        """
        if self._experiment:
            return self._experiment

        os.makedirs(self.save_dir, exist_ok=True)
        self._experiment = ExperimentWriter(log_dir=self.save_dir)
        return self._experiment

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = self._convert_params(params)
        self.experiment.log_hparams(params)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        metrics = self._add_prefix(metrics)
        self.experiment.log_metrics(metrics, step)

    @rank_zero_only
    def save(self) -> None:
        super().save()
        self.experiment.save()

    @rank_zero_only
    def finalize(self, status: str) -> None:
        self.save()

    @property
    def name(self):
        pass

    @property
    def version(self):
        pass


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
        self, params: Union[Dict[str, Any], Namespace], metrics: Optional[Dict[str, Any]] = None
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
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
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
