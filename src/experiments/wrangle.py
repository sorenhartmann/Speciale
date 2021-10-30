from omegaconf import OmegaConf
from src.experiments.common import MultiRun, Run
from functools import cache
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd

@cache
def get_run_statistics(run: Run, tag="err/val"):

    tag = "err/val"

    dir_ = run.path / "metrics"
    if not dir_.exists():
        return None

    accumulator = EventAccumulator(str(run.path / "metrics"))
    accumulator.Reload()

    index = [x.step for x in accumulator.Scalars(tag)]
    values = [x.value for x in accumulator.Scalars(tag)]

    statistics = pd.Series(values, name=tag, index=pd.Index(index, name="step"))
    return statistics


def get_multirun_statistics(multirun: MultiRun, config_values=[], tag="err/val"):

    val_errs = {run.id: get_run_statistics(run, tag) for run in multirun.runs}

    def get_config(run):
        return {
            config: OmegaConf.select(run.config, config) for config in config_values
        }

    configs = {run.id: get_config(run) for run in multirun.runs}

    metrics = (
        pd.concat(val_errs, names=["id"])
        .pipe(pd.DataFrame)
        .sort_index()
        .join(
            pd.DataFrame.from_dict(configs, orient="index").rename_axis(index="id"),
            how="right",
        )
    )
    return metrics