from functools import cache, wraps

import matplotlib.pyplot as plt
import pandas as pd
from pandas.core.indexes.base import Index
import seaborn as sns
from omegaconf import OmegaConf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.experiments.common import MultiRun, Run, cast_run, set_directory
from pathlib import Path
import torch
import numpy as np


@cache
def load_estimates(run: Run, sub_dir, single_step=None):

    values = []
    step = []

    files = sorted((run.path / sub_dir).iterdir())
    if single_step is not None:
        files = [files[single_step]]

    for file in files:
        values.append(torch.load(file))
        step.append(int(file.stem))

    sampled_idx = torch.load(run.path / "log_idx.pt").numpy()

    return (
        pd.DataFrame(
            torch.stack(values).numpy(),
            pd.Index(step, name="step"),
            columns=sampled_idx,
        )
        .reset_index()
        .melt(
            id_vars="step",
            var_name="parameter_index",
            value_name=sub_dir,
        )
        .set_index(["parameter_index", "step"])
    )


@cast_run
def grad_variance_estimates(multirun: MultiRun):

    def get_estimates(run):
        variance_inter_batch = load_estimates(run, "variance_inter_batch", -1)
        variance_estimated = load_estimates(run, "variance_estimated", -1)
        return pd.concat([variance_inter_batch, variance_estimated], axis=1)

    estimates = {run.run_index: get_estimates(run) for run in multirun.runs}

    def get_params(run):
        estimator_config = run.config.inference.sampler.variance_estimator
        estimator = run.config.variance_estimator._target_.split(".")[-1]
        adj_w_mean = getattr(run.config.variance_estimator, "adj_with_mean", None)
        if adj_w_mean is not None:
            estimator += f"({adj_w_mean=})"
        return {
            "use_estimate": estimator_config.use_estimate,
            "estimator": estimator,
        }

    params = {run.run_index: get_params(run) for run in multirun.runs}

    fg = (
        pd.concat(x.assign(i=i) for i, x in estimates.items())
        .set_index("i", append=True)
        .reorder_levels(["i", "parameter_index", "step"])
        .join(
            pd.DataFrame.from_dict(params, orient="index").rename_axis("i"),
            how="left",
        )
        .pipe(
            (sns.relplot, "data"),
            x="variance_inter_batch",
            y="variance_estimated",
            col="estimator",
            row="use_estimate",
        )
    )
    fg.set(xscale="log")
    fg.set(yscale="log")
    for ax in fg.axes.flatten():
        ax.axline((0, 0), (1, 1), color="C1")

    plt.savefig("final_estimates.pdf")


@cast_run
def all_estimates_sampled_variables(run: Run, n_parameters=9, seed=123):

    variance_inter_batch = load_estimates(run, "variance_inter_batch")
    variance_estimated = load_estimates(run, "variance_estimated")
    # fmt: off
    data = (
        pd.concat([variance_inter_batch, variance_estimated], axis=1)
        .unstack("parameter_index")
    )
    is_zero = (
        np.isclose(data["variance_inter_batch"], 0).all(0) 
        |  np.isclose(data["variance_inter_batch"], 0).all(0)
    )
    # fmt: on
    no_zero_columns = (
        data.columns.get_level_values("parameter_index").to_series().unique()[~is_zero]
    )
    sampled_cols = pd.Series(no_zero_columns).sample(n_parameters, random_state=seed)

    sampled_data = (
        data.reorder_levels((1, 0), axis=1)
        .loc[:, sampled_cols]
        .stack(level="parameter_index")
        .reset_index()
        .sample(frac=1.0)
    )
    fg = sns.relplot(
        data=sampled_data,
        x="variance_inter_batch",
        y="variance_estimated",
        col="parameter_index",
        col_wrap=3,
        facet_kws={"sharey": False, "sharex": False},
    )

    fg.set(xscale="log")
    fg.set(yscale="log")
    for ax in fg.axes.flatten():
        ax.axline((0, 0), (1, 1), color="red")

    plt.savefig("all_estimates.pdf")
