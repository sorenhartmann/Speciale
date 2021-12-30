from typing import Any, Dict, Optional, Tuple, List, Union, overload
from matplotlib.contour import QuadContourSet
from torch import Tensor
from torch.distributions import MultivariateNormal, Normal
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns


def get_exact_posterior(X: Tensor, Y: Tensor) -> MultivariateNormal:

    m = X.shape[1]
    ols = (X.T @ X).inverse() @ X.T @ Y.squeeze()

    L_0 = torch.eye(m)
    mu_0 = torch.zeros(m)
    L_n = X.T @ X + L_0
    mu_n = L_n.inverse() @ (X.T @ X @ ols + L_0 @ mu_0)
    posterior = torch.distributions.MultivariateNormal(mu_n, precision_matrix=L_n)
    return posterior


@overload
def get_marginal(dist: MultivariateNormal, i: int) -> Normal:
    ...


@overload
def get_marginal(dist: MultivariateNormal, i: int, j: int) -> MultivariateNormal:
    ...


def get_marginal(
    dist: MultivariateNormal,
    i: int,
    j: Optional[int] = None,
) -> Union[MultivariateNormal, Normal]:

    if j is None:
        mean = dist.mean[i]
        var = dist.covariance_matrix[i, i]
        return Normal(mean, var.sqrt())
    else:
        mean = dist.mean[[i, j]]
        v = dist.covariance_matrix
        cov = torch.tensor(
            [
                [v[i, i], v[i, j]],
                [v[j, i], v[j, j]],
            ]
        )
        return MultivariateNormal(mean, cov)


LIM_TYPE = Tuple[float, float]


def get_lims(mean: float, stddev: float) -> LIM_TYPE:
    lims = mean + torch.tensor([-5, 5]) * stddev
    return (lims[0].item(), lims[1].item())


def draw_uni_gaussian(
    dist: Normal,
    xlim: LIM_TYPE = None,
    transpose: bool = False,
    **kwargs: Any,
) -> List[Line2D]:

    xlim = xlim if xlim is not None else get_lims(dist.mean, dist.stddev)

    xx = torch.linspace(*xlim, steps=300)
    yy = dist.log_prob(xx).exp()
    if transpose:
        xx, yy = yy, xx

    return plt.plot(xx, yy, **kwargs)


def draw_bi_gaussian(
    dist: MultivariateNormal,
    xlim: Optional[LIM_TYPE] = None,
    ylim: Optional[LIM_TYPE] = None,
    **kwargs: Any,
) -> QuadContourSet:

    xlim = xlim if xlim is not None else get_lims(dist.mean[0], dist.stddev[0])
    ylim = ylim if ylim is not None else get_lims(dist.mean[1], dist.stddev[1])

    xx = torch.linspace(*xlim, 300)
    yy = torch.linspace(*ylim, 300)
    XY = torch.stack(torch.meshgrid(xx, yy), dim=-1)
    ZZ = dist.log_prob(XY).exp()
    plt.contour(XY[..., 0], XY[..., 1], ZZ, **kwargs)


# For use with sns facet
def sns_facet__plot_univariate(
    x: pd.Series,
    posterior: MultivariateNormal,
    transpose: bool = False,
    lims: Optional[LIM_TYPE] = None,
    **kwargs: Any,
) -> None:

    i = int(x.name)
    marg = get_marginal(posterior, i)
    xlims = lims if lims is not None else (x.min(), x.max())
    draw_uni_gaussian(marg, xlim=xlims, transpose=transpose, c="black")


# For use with sns facet
def sns_facet__plot_bivariate(
    x: pd.Series,
    y: pd.Series,
    posterior: MultivariateNormal,
    xlims: Optional[LIM_TYPE] = None,
    ylims: Optional[LIM_TYPE] = None,
    levels: Optional[List[float]] = None,
    **kwargs: Any,
) -> None:

    i = int(x.name)
    j = int(y.name)
    marg = get_marginal(posterior, i, j)
    xlims = (min(x), max(x)) if xlims is None else xlims
    ylims = (min(y), max(y)) if ylims is None else ylims
    xx = torch.linspace(*xlims, 400)
    yy = torch.linspace(*ylims, 400)
    XY = torch.stack(torch.meshgrid(xx, yy), dim=-1)
    ZZ = marg.log_prob(XY).exp()
    plt.contour(XY[..., 0], XY[..., 1], ZZ, colors="black", levels=levels)


def plot_sampled_distributions_pairs(
    sample_data: pd.DataFrame,
    exact_posterior: MultivariateNormal,
    color: Optional[Any] = None,
) -> sns.PairGrid:

    fg: sns.PairGrid = sns.pairplot(
        sample_data,
        kind="hist",
        diag_kws={"stat": "density", "bins": 50, "rasterized": True, "color": color},
        plot_kws={"bins": 50, "color": color},
    )
    fg.map_diag(sns_facet__plot_univariate, posterior=exact_posterior)
    fg.map_offdiag(sns_facet__plot_bivariate, posterior=exact_posterior)

    # Add labels
    for ax in fg.axes.flat:
        xlabel: str = ax.get_xlabel()
        if xlabel.isdigit():
            ax.set_xlabel(f"$a_{xlabel}$")
        ylabel: str = ax.get_ylabel()
        if ylabel.isdigit():
            ax.set_ylabel(f"$a_{ylabel}$")

    return fg


def plot_sampled_joint_bivariate(
    sample_data: pd.DataFrame,
    exact_posterior: MultivariateNormal,
    i: int,
    j: int,
    xlims: Optional[LIM_TYPE] = None,
    ylims: Optional[LIM_TYPE] = None,
    levels: Optional[List[float]] = None,
    **kwargs: Any,
) -> sns.JointGrid:

    x = sample_data.iloc[:, i]
    y = sample_data.iloc[:, j]

    xlims = (min(x), max(x)) if xlims is None else xlims
    ylims = (min(y), max(y)) if ylims is None else ylims
    levels = levels if levels is not None else [1e-6, 1e-4, 1e-2, 1e0]

    fg: sns.JointGrid = sns.jointplot(
        x=x,
        y=y,
        kind="hist",
        marginal_kws={"stat": "density", "bins": 40},
        height=4,
        bins=40,
        binrange=(xlims, ylims),
        **kwargs,
    )

    plt.sca(fg.ax_joint)
    sns_facet__plot_bivariate(
        x,
        y,
        posterior=exact_posterior,
        xlims=xlims,
        ylims=ylims,
        levels=levels,
    )
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.sca(fg.ax_marg_x)
    sns_facet__plot_univariate(x, posterior=exact_posterior, lims=xlims)
    plt.sca(fg.ax_marg_y)
    sns_facet__plot_univariate(y, posterior=exact_posterior, transpose=True, lims=ylims)
    
    fg.set_axis_labels(xlabel=f"$a_{i}$", ylabel=f"$a_{j}$")
