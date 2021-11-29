import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


PLOT_COLORS = {
    ("HMC", 15): {"color_palette": "Blues_r"},
    ("HMC", 5): {"color_palette": "Greens_r"},
    ("SGHMC", 5): {"color_palette": "Oranges_r"},
    ("SGHMCWithVarianceEstimator", 5): {"color_palette": "RdPu_r"},
}


def load_samples(run):
    def to_matrix(samples):
        return torch.stack(list(samples.values())).numpy()

    return (
        pd.DataFrame(to_matrix(torch.load(run.path / "saved_samples.pt")))
        .rename_axis(index="sample")
        .assign(
            sampler=run.config["inference"]["sampler"]["_target_"],
            batch_size=run.config["data"]["batch_size"],
        )
        .assign(
            sampler=lambda x: x.sampler.str.extract(r"src.inference.mcmc.samplers.(.+)")
        )
        .set_index(["sampler", "batch_size"], append=True)
        .reorder_levels(["sampler", "batch_size", "sample"])
    )


def get_exact_posterior(X: torch.Tensor, Y: torch.Tensor):

    m = X.shape[1]
    ols = (X.T @ X).inverse() @ X.T @ Y.squeeze()

    L_0 = torch.eye(m)
    mu_0 = torch.zeros(m)
    L_n = X.T @ X + L_0
    mu_n = L_n.inverse() @ (X.T @ X @ ols + L_0 @ mu_0)
    posterior = torch.distributions.MultivariateNormal(mu_n, precision_matrix=L_n)
    return posterior


def get_marginal(dist: torch.distributions.Normal, i, j=None):

    if j is None:
        mean = dist.mean[i]
        var = dist.covariance_matrix[i, i]
        return torch.distributions.Normal(mean, var.sqrt())
    else:
        mean = dist.mean[[i, j]]
        v = dist.covariance_matrix
        cov = torch.tensor(
            [
                [v[i, i], v[i, j]],
                [v[j, i], v[j, j]],
            ]
        )
        return torch.distributions.MultivariateNormal(mean, cov)


# For use with sns pairplot
def plot_univariate(x, posterior, transpose=False, lims=None, **kwargs):
    i = x.name
    marg = get_marginal(posterior, i)
    if lims is None:
        xlims = min(x), max(x)
    else:
        xlims = lims
    xx = torch.linspace(*xlims)
    yy = marg.log_prob(xx).exp()
    if transpose:
        xx, yy, = (
            yy,
            xx,
        )
    plt.plot(xx, yy, c="black")


# For use with sns pairplot
def plot_bivariate(x, y, posterior, xlims=None, ylims=None, levels=None, **kwargs):
    i = x.name
    j = y.name
    marg = get_marginal(posterior, i, j)
    xlims = (min(x), max(x)) if xlims is None else xlims
    ylims = (min(y), max(y)) if ylims is None else ylims
    xx = torch.linspace(*xlims, 400)
    yy = torch.linspace(*ylims, 400)
    XY = torch.stack(torch.meshgrid(xx, yy), dim=-1)
    ZZ = marg.log_prob(XY).exp()
    plt.contour(XY[..., 0], XY[..., 1], ZZ, colors="black", levels=levels)


def plot_sampled_distributions_pairs(sample_data, exact_posterior):
    fg = sns.pairplot(
        sample_data,
        kind="hist",
        diag_kws={"stat": "density", "bins": 50, "rasterized": True},
        plot_kws={"bins": 50},
    )
    fg.map_diag(plot_univariate, posterior=exact_posterior)
    fg.map_offdiag(plot_bivariate, posterior=exact_posterior)
    return fg


def plot_sampled_joint_bivariate(
    sample_data,
    exact_posterior,
    xlims=None,
    ylims=None,
):

    x = sample_data.iloc[:, 0]
    y = sample_data.iloc[:, 1]

    fg = sns.jointplot(
        x=x,
        y=y,
        kind="hist",
        marginal_kws={"stat": "density", "bins": 40},
        height=4,
        bins=40,
        binrange=(xlims, ylims),
    )
    plt.sca(fg.ax_joint)
    plot_bivariate(
        x,
        y,
        posterior=exact_posterior,
        xlims=xlims,
        ylims=ylims,
        levels=[1e-6, 1e-4, 1e-2, 1e0],
    )
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.sca(fg.ax_marg_x)
    plot_univariate(x, posterior=exact_posterior, lims=xlims)
    plt.sca(fg.ax_marg_y)
    plot_univariate(y, posterior=exact_posterior, transpose=True, lims=ylims)
