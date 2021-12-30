from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import chi2
from src.utils import Run

from .temperatures import plot_temperature_chi2
from .colors import get_colors


def plot_val_err(
    runs: List[Run],
    labels: Optional[List[str]] = None,
    **plot_kwargs: Any,
) -> sns.FacetGrid:

    if labels is None:
        labels = [r.inference_label for r in runs]

    palette, hue_order = get_colors(dict(zip(labels, runs)))

    relplot_kwargs = {"height": 3.2, "aspect": 1.8}
    relplot_kwargs.update(plot_kwargs)

    return (
        pd.concat(
            pd.DataFrame(
                {
                    "Validation error": r.get_scalar("err/val"),
                    "Epoch": r.get_scalar("epoch"),
                }
            ).assign(Method=l)
            for l, r in zip(labels, runs)
        )
        .dropna()
        .set_index("Method", append=True)
        .reorder_levels(["Method", "step"])
        .sort_index()
        .reset_index()
        .drop_duplicates(["Method", "Epoch"], keep="last")
        .pipe(
            (sns.relplot, "data"),
            x="Epoch",
            y="Validation error",
            kind="line",
            hue="Method",
            palette=palette,
            hue_order=hue_order,
            **relplot_kwargs,
        )
    )


def plot_mcmc_downsampling(
    runs: List[Run],
    labels: Optional[List[str]] = None,
    **plot_kwargs: Any,
) -> sns.FacetGrid:

    if labels is None:
        labels = [r.inference_label for r in runs]

    palette, hue_order = get_colors(dict(zip(labels, runs)))

    mcmc_labelled_runs = {
        l: r for l, r in zip(labels, runs) if "SGHMC" in r.inference_label
    }
    other_labelled_runs = {
        l: r for l, r in zip(labels, runs) if "SGHMC" not in r.inference_label
    }

    relplot_kwargs = {"height": 3.2, "aspect": 1.8}
    relplot_kwargs.update(plot_kwargs)

    return (
        pd.concat(
            pd.read_json(r._dir / "sample_resampling_curve.json")
            .rename_axis(index=["n_sampled"])
            .assign(sampler=l)
            .set_index("sampler", append=True)
            .reorder_levels(["sampler", "n_sampled"])
            .sort_index()
            for l, r in mcmc_labelled_runs.items()
        )
        .unstack("sampler")
        .droplevel(0, axis=1)
        .assign(
            **{
                l: r.get_scalar("err/test").item()
                for l, r in other_labelled_runs.items()
            }
        )
        .stack()
        .rename("Test error")
        .reset_index()
        .sort_values(["sampler", "n_sampled"])
        .rename(columns={"n_sampled": "$K$ MCMC Samples", "sampler": "Method"})
        .pipe(
            (sns.relplot, "data"),
            x="$K$ MCMC Samples",
            y="Test error",
            hue="Method",
            kind="line",
            **relplot_kwargs,
            palette=palette,
            hue_order=hue_order,
        )
    )


def plot_calibration(
    runs: List[Run],
    labels: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (8, 4),
    legend_cols: int = 3,
) -> None:

    if labels is None:
        labels = [r.inference_label for r in runs]

    palette, hue_order = get_colors(dict(zip(labels, runs)))

    data = (
        pd.concat(
            pd.read_csv(r._dir / "ce_stats.csv", index_col=0).assign(inference=l)
            for l, r in zip(labels, runs)
        )
        .loc[lambda x: x["count"] > 5]
        .reset_index()
    )

    kwargs = {
        "data": data,
        "x": "mean_confidence",
        "y": "mean_accuracy",
        "hue": "inference",
        "hue_order": hue_order,
        "palette": palette,
    }

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for i, lims in enumerate([(0, 1), (0.95, 1.0)]):

        plt.sca(axes[i])

        plt.axline((0, 0), (1, 1), color="grey", linestyle="dashed")
        sns.scatterplot(**kwargs, legend=False)
        l = sns.lineplot(**kwargs, legend=i == 1)
        sns.despine()
        plt.xlabel("Confidence")
        if i == 0:
            plt.ylabel("Accuracy")
        else:
            plt.gca().set_ylabel(None)
        plt.gca().axis("equal")
        plt.xlim(lims)
        plt.ylim(lims)

    lines = l.legend_.get_lines()
    texts = [t.get_text() for t in l.legend_.texts]
    l.legend_.remove()  # Remove seaborn legends
    plt.tight_layout()
    plt.figlegend(
        lines,
        texts,
        title="Method",
        ncol=legend_cols,
        loc="upper center",
        frameon=False,
    )
    plt.subplots_adjust(top=0.75)


def plot_temperatures(
    runs: List[Run],
    labels: Optional[List[str]] = None,
    **plot_kwargs: Any,
) -> sns.FacetGrid:

    if labels is None:
        labels = [r.inference_label for r in runs]

    mcmc_labelled_runs = {
        l: r for l, r in zip(labels, runs) if "SGHMC" in r.inference_label
    }

    runs = list(mcmc_labelled_runs.values())
    labels = list(mcmc_labelled_runs.keys())

    palette, hue_order = get_colors(dict(zip(labels, runs)))

    relplot_kwargs = {
        "height": 1.8,
        "aspect": 2,
        "col_wrap": 2,
    }
    relplot_kwargs.update(plot_kwargs)

    temperatures = pd.concat(
        pd.DataFrame.from_dict(
            torch.load(run._dir / "temperature_samples.pt"),
            orient="index",
        )
        .rename_axis(index=["step", "parameter"])
        .loc[lambda x: x.index.get_level_values("step") % 50 == 0]
        .assign(Sampler=l)
        .set_index("Sampler", append=True)
        .reorder_levels(["Sampler", "parameter", "step"])
        for l, run in zip(labels, runs)
    )

    n_params = temperatures.index.get_level_values("parameter").nunique()
    if n_params > 10:
        sampled_params = (
            temperatures.index.get_level_values("parameter")
            .unique()
            .to_series()
            .sample(10, random_state=42)
            .sort_values()
            .values
        )

        temperatures = temperatures.loc[
            lambda x: x.index.get_level_values("parameter").isin(sampled_params)
        ]


    fg = sns.displot(
        data=temperatures.reset_index(),
        x="temperature_sum",
        hue="Sampler",
        kind="kde",
        col="parameter",
        palette=palette,
        hue_order=hue_order,
        common_norm=False,
        facet_kws={"sharex": False, "sharey": False},
        **relplot_kwargs,
    )

    lines, texts = plot_temperature_chi2(fg, linestyle="--")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2, right=1)

    plt.figlegend(
        lines,
        texts,
        title="Sampler",
        ncol=3,
        frameon=False,
        loc="lower center",
    )

    return fg
