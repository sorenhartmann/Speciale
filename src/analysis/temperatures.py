from scipy.stats import chi2

import seaborn as sns
from typing import Dict, Iterable, List, Any, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

import pandas as pd
from dataclasses import dataclass
from matplotlib.lines import Line2D
import torch
from src.utils import Run


def load_temperatures(run: Run) -> pd.DataFrame:
    return (
        pd.DataFrame.from_dict(
            torch.load(run._dir / "temperature_samples.pt"),
            orient="index",
        )
        .rename_axis(index=["step", "parameter"])
        .reorder_levels(["parameter", "step"])
    )


@dataclass
class Pointer:
    to: Optional[Any] = None


def plot_chi2_sub(
    df: pd.DataFrame, line_ref: Pointer, color: Optional[Any] = None, **kwargs: Any
) -> None:

    xlim: Tuple[float, float] = plt.gca().axes.get_xlim()

    xx = np.linspace(*xlim, 300)
    yy = chi2(df.iloc[0]).pdf(xx)
    line_ref.to = plt.plot(xx, yy, color="black", **kwargs)


def plot_temperature_chi2(
    fg: sns.FacetGrid,
    **kwargs: Any,
) -> Tuple[List[Line2D], List[str]]:
    """
    Example:
    >>> fg = sns.displot(data=temperature_samples, ...)
    >>> lines, texts = plot_temperature_chi2(fg)
    >>> plt.legend(lines, texts , ...)
    """

    true_line = Pointer()

    fg.map(plot_chi2_sub, "n_params", line_ref=true_line, **kwargs)

    for ax in fg.axes:
        if ax.get_xlabel():
            ax.set_xlabel("$\hat{T}_K\cdot d$")

    lines = true_line.to
    texts = ["$\chi^2(d)$"]

    try:
        lines = fg.legend.get_lines() + lines
        texts = [t.get_text() for t in fg.legend.texts] + texts
        fg.legend.remove()  # Remove seaborn legends
    except AttributeError:
        pass

    return lines, texts


def get_frac_in_ci(
    temperature_samples: pd.DataFrame,
    groupby_levels: List[str],
    c: float = 0.99,
) -> pd.DataFrame:

    conf_ints = pd.DataFrame.from_dict(
        {
            d: {
                "lower": chi2(d).ppf((1 - c) / 2) / d,
                "upper": chi2(d).ppf((1 + c) / 2) / d,
            }
            for d in temperature_samples.n_params.unique()
        },
        orient="index",
    ).rename_axis(index="n_params")

    return (
        temperature_samples.join(conf_ints, on="n_params")
        .assign(is_in_ci=lambda x: (x.T_k > x.lower) & (x.T_k < x.upper))
        .is_in_ci.groupby(level=groupby_levels)
        .agg(frac_in_ci="mean", count="count")
    )
