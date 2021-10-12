import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from hydra.utils import instantiate
from tqdm import trange

from src.experiments.common import Experiment
from src.inference.mcmc.samplers import Samplable

sns.set_style("whitegrid")


class Example(Samplable):

    state = torch.tensor(0.0)

    def __init__(self, grad_noise=0):
        self.grad_noise = grad_noise

    def prop_log_p(self):
        return 2 * self.state ** 2 - self.state ** 4

    def grad_prop_log_p(self):
        value = 4 * self.state - 4 * self.state ** 3
        if self.grad_noise > 0.0:
            value += torch.randn_like(value)
        return value

    @classmethod
    def constant(cls):

        from math import exp, pi

        from scipy.special import iv

        return torch.tensor(
            1 / 2 * exp(1 / 2) * pi * (iv(-1 / 4, 1 / 2) + iv(1 / 4, 1 / 2))
        )

    @classmethod
    def density(cls, x):

        return np.exp(2 * x ** 2 - x ** 4) / cls.constant()


@hydra.main("../../conf", "experiment/synthetic/config")
def experiment(cfg):

    distribution = instantiate(cfg.distribution)
    sampler = instantiate(cfg.sampler)
    sampler.setup(distribution)

    samples = torch.empty((cfg.n_samples,))
    for i in trange(cfg.n_samples):
        samples[i] = sampler.next_sample()

    torch.save(samples, "samples.pt")

    if cfg.plot_distribution:
        xx = torch.linspace(-3, 3, 200)
        plt.plot(xx, distribution.density(xx))
        _plot_dist(samples, plt.gca())
        plt.show()

def _plot_dist(samples, bins, *args, ax=None, **kwargs):

    ax = kwargs.pop("ax", None)
    if ax is None:
        ax = plt.gca()

    xx = bins[:-1] + (bins[1] - bins[0]) / 2
    yy, _ = np.histogram(samples, bins, density=True)
    ax.plot(xx, yy, *args, **kwargs)

def plot_distribution_comparison(experiment: Experiment, ax=None):

    if ax is None:
        ax = plt.gca()

    n_bins = 60
    bins = np.linspace(-3, 3, n_bins + 1)

    xx = np.linspace(bins[0], bins[-1], 200)

    sns.lineplot(
        x="x",
        y="density",
        style="legend",
        data=(
            experiment.as_dataframe()
            .groupby("legend")
            .last()
            .path.apply(lambda x: torch.load(x / "samples.pt"))
            .map(lambda x: np.histogram(x, bins=bins, density=True)[0])
            .apply(pd.Series)
            .transpose()
            .assign(x=bins[:-1] + (bins[1] - bins[0]) / 2)
            .melt(id_vars="x", value_name="density")
        ),
        hue="legend",
        ax=ax,
    )
    ax.plot(xx, Example.density(xx), label="True distribution", color="grey")

    ax.set_ylim((0, 0.9))
    ax.set_xlim((-2, 2))

    ax.legend()
    plt.show()

if __name__ == "__main__":

    experiment()
