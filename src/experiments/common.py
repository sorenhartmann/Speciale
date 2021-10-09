import os
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from omegaconf import Container, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.saving import save_hparams_to_yaml
from pytorch_lightning.loggers.base import (LightningLoggerBase,
                                            rank_zero_experiment)
from pytorch_lightning.loggers.csv_logs import ExperimentWriter
from pytorch_lightning.utilities import (_OMEGACONF_AVAILABLE, rank_zero_only,
                                         rank_zero_warn)
from pytorch_lightning.utilities.cloud_io import get_filesystem
from pytorch_lightning.utilities.distributed import rank_zero_only
# from src.utils import Component, HPARAM
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

ROOT_DIR = Path(__file__).parents[2]

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



def default_callbacks():
    
    return [ModelCheckpoint("./checkpoints")]
