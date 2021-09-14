from abc import abstractmethod, ABC
from contextlib import contextmanager
import contextlib
import datetime
import inspect
import pickle
import os
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
from src.modules import BayesianModel
from src.inference import BayesianRegressor
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
from pytorch_lightning.utilities.distributed import rank_zero_only
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

from argparse import Namespace
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).parents[2]


class GetSampler(argparse.Action):

    default = Hamiltonian
    samplers = samplers

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, self.samplers[values])


def get_args(model_cls: Type[BayesianModel], inference_cls: Type[BayesianRegressor]):

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


# class Experiment(ABC):

#     def register_configuration(self, configuration):
#         self.configuration = configuration

#     def get_configuration(self):
#         conf = getattr(self, "configuration", None)
#         if conf is None:
#             attrs = inspect.signature(self.__init__).parameters.keys()
#             conf = {attr: getattr(self, attr) for attr in attrs}

#         return conf

#     @classmethod
#     @property
#     def name(cls):
#         return cls.__name__

#     @classmethod
#     @property
#     def path(cls):
#         return ROOT_DIR / "experiment_results" / cls.name

#     @classmethod
#     @contextmanager
#     def shelve(cls):
#         with shelve.open(str(cls.path), writeback=True) as s:
#             try:
#                 yield s
#             finally:
#                 pass

#     @classmethod
#     def runs(cls):
#         with cls.shelve() as s:
#             runs = s["runs"]
#         return runs


#     @abstractmethod
#     def experiment(self):
#         pass

#     def run(self, log=True):

#         time = datetime.datetime.now()
#         configuration = self.get_configuration()
#         results = self.experiment()

#         if not log:
#             return results

#         with self.shelve() as s:

#             if "runs" not in s:
#                 s["runs"] = []

#             s["runs"].append(
#                 {"time": time, "configuration": configuration, "results": results}
#             )

#         return results

#     @classmethod
#     def plots(cls, i=None):

#         if i is None:
#             i = -1

#         run = cls.runs()[i]

#         return [
#             getattr(cls, method)(run)
#             for method in dir(cls)
#             if method.startswith("plot_")
#         ]



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
