from functools import cache
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from omegaconf import OmegaConf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.experiments.common import MultiRun, Run, cast_run

from .parallel_coordinates import parallel_coordinates


@cache
def get_val_err(run, n=10, tag="err/val"):
    acc = EventAccumulator(str(run.path / "metrics"))
    acc.Reload()
    return sum(x.value for x in acc.Scalars(tag)[-n:]) / n


@cast_run
def parameter_sweep(multi_run: MultiRun, parameters, tag):

    val_errs = {run.run_index: get_val_err(run) for run in multi_run.runs}

    def get_params(run):
        return {param: OmegaConf.select(run.config, param) for param in parameters}

    param_values = {run.run_index: get_params(run) for run in multi_run.runs}

    metrics = (
        pd.DataFrame.from_dict(param_values, orient="index")
        .sort_index()
        .assign(**{tag: pd.Series(val_errs)})
    )

    plt.figure(figsize=(12, 6))
    parallel_coordinates(metrics, parameters, tag, cmap="plasma")

    plt.savefig("parameter_sweep.pdf")


@cache
def get_run_statistics(run: Run, tag="err/val"):

    accumulator = EventAccumulator(str(run.path / "metrics"))
    accumulator.Reload()

    def get_series(tag):
        index = [x.step for x in accumulator.Scalars(tag)]
        value = [x.value for x in accumulator.Scalars(tag)]
        return pd.Series(value, index=index)

    return pd.DataFrame(
        {
            "Epoch": get_series("epoch"),
            tag: get_series(tag),
        }
    )


@cast_run
def validation_curves(multi_run: MultiRun, y_val="err/val", **relplot_semantics):
    def get_params(run):
        return {
            param: OmegaConf.select(run.config, param)
            for param in relplot_semantics.values()
        }

    param_values = {run.run_index: get_params(run) for run in multi_run.runs}

    statistics = {
        run.run_index: get_run_statistics(run, y_val) for run in multi_run.runs
    }

    (
        pd.concat(x.assign(i=i) for i, x in statistics.items())
        .set_index("i", append=True)
        .join(
            pd.DataFrame.from_dict(param_values, orient="index").rename_axis("i"),
            how="left",
        )
        .pipe(
            (sns.relplot, "data"), x="Epoch", y=y_val, kind="line", **relplot_semantics
        )
    )

    plt.savefig("validation_curves.pdf")
