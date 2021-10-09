import functools
from math import exp, pi

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.special import iv

from src.experiments.common import ROOT_DIR, ExperimentHandler, Run
from src.samplers import (Hamiltonian, HamiltonianNoMH, Samplable,
                          StochasticGradientHamiltonian)


@functools.cache
def constant():
    return torch.tensor(
        1 / 2 * exp(1 / 2) * pi * (iv(-1 / 4, 1 / 2) + iv(1 / 4, 1 / 2))
    )


def p_exact(x):
    return (2 * x ** 2 - x ** 4).exp() / constant()


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


def experiment(
    n_samples=80_000,
    step_size=0.1,
    grad_noise=2.0,
    n_steps=50,
):

    wo_noise = Example()
    w_noise = Example(grad_noise=grad_noise)

    hmc = Hamiltonian(step_size=step_size, n_steps=n_steps)
    hmc_wo_acc_step = HamiltonianNoMH(step_size=step_size, n_steps=n_steps)
    sghmc = StochasticGradientHamiltonian(
        step_size=step_size,
        n_steps=n_steps,
        M=1.0,
        C=3.0,
        V=grad_noise ** 2,
    )

    def get_samples(sampler, samplable):
        samples = torch.empty((n_samples,))
        sampler.setup(samplable)
        for i in range(n_samples):
            samples[i] = sampler.next_sample()

        return samples

    results = {}

    results["hmc"] = get_samples(hmc, wo_noise)
    print("Experiment done!")
    results["hmc_no_acc_step"] = get_samples(hmc_wo_acc_step, wo_noise)
    print("Experiment done!")
    results["hmc_w_noise"] = get_samples(hmc, w_noise)
    print("Experiment done!")
    results["hmc_wo_acc_step_w_noise"] = get_samples(hmc_wo_acc_step, w_noise)
    print("Experiment done!")
    results["sghmc"] = get_samples(sghmc, w_noise)
    print("Experiment done!")

    return pd.DataFrame(results)

def _plot_dist(samples, bins, *args, **kwargs):

    ax = kwargs.pop("ax")
    if ax is None:
        ax = plt.gca()

    xx = bins[:-1] + (bins[1] - bins[0]) / 2
    yy, _ = np.histogram(samples, bins, density=True)
    ax.plot(xx, yy, *args, **kwargs)


def plot_distribution(run: Run, ax=None):

    n_bins = 60

    if ax is None:
        ax = plt.gca()

    result = run.result

    xx = torch.linspace(-3, 3, 200)
    ax.plot(xx, p_exact(xx), label="True distribution")

    bins = np.linspace(-3, 3, n_bins + 1)
    xx = bins[:-1] + (bins[1] - bins[0]) / 2

    _plot_dist(result["hmc"], bins, "-v", fillstyle="none", label="HMC", ax=ax)
    _plot_dist(result["hmc_no_acc_step"], bins, "-.", label="HMC (No MH)", ax=ax)
    _plot_dist(result["hmc_w_noise"], bins, "-x", label="HMC /w noise", ax=ax)
    _plot_dist(
        result["hmc_wo_acc_step_w_noise"],
        bins,
        "-.",
        label="HMC /w noise (No MH)",
        ax=ax,
    )
    _plot_dist(result["sghmc"], bins, "-", label="SGHMC", ax=ax)

    ax.set_xlim(-2, 2)
    ax.legend()


def main():

    handler = ExperimentHandler(experiment)
    # handler.run()
    run = handler.latest_run()

    plot_distribution(run)


if __name__ == "__main__":

    main()
