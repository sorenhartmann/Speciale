from torch import nn
from torch.autograd import grad
from src.modules import BayesianModel
import matplotlib.pyplot as plt
import torch
from scipy.special import iv
from math import pi, exp, sqrt
import functools
from src.samplers import Hamiltonian, MetropolisHastings, Samplable, clone_parameters
import seaborn as sns
import numpy as np
from tqdm import trange


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

        self.state.normal_()
        self.grad_noise = grad_noise

    def prop_log_p(self):
        return 2 * self.state ** 2 - self.state ** 4

    def grad_prop_log_p(self):
        value = 4 * self.state - 4 * self.state ** 3
        if self.grad_noise > 0.0:
            value += torch.randn_like(value)
        return value


class HamiltonianNoMH(Hamiltonian):
    def next_sample(self, *args):

        self.resample_momentum()
        self.step_momentum(*args, half_step=True)
        for i in range(self.n_steps):
            self.step_parameters()
            self.step_momentum(*args, half_step=(i == self.n_steps - 1))

        return self.samplable.state.clone()


def plot_dist(samples, bins, *args, **kwargs):
    xx = bins[:-1] + (bins[1] - bins[0]) / 2
    yy, _ = np.histogram(samples, bins, density=True)
    plt.plot(xx, yy, *args, **kwargs)


def main():

    n_samples = 80_000
    n_bins = 60
    step_size = 0.1
    grad_noise = 2.0
    n_steps = 50

    bins = torch.linspace(-3, 3, n_bins + 1)
    plt.plot(bins, p_exact(bins), label="True distribution")

    def experiment(sampler_cls, samplable, *args, **kwargs):
        samples = torch.empty((n_samples,))
        sampler = sampler_cls(step_size=step_size, n_steps=n_steps).setup(samplable)
        for i in trange(n_samples):
            samples[i] = sampler.next_sample()
        plot_dist(samples, bins, *args, **kwargs)

    experiment(Hamiltonian, Example(), "-v", fillstyle="none", label="HMC")
    experiment(HamiltonianNoMH, Example(), "-.", label="HMC (No MH)")
    experiment(Hamiltonian, Example(grad_noise=grad_noise), "-x", label="HMC /w noise")
    experiment(
        HamiltonianNoMH,
        Example(grad_noise=grad_noise),
        "-.",
        label="HMC /w noise (No MH)",
    )

    plt.legend()
    plt.show()


if __name__ == "__main__":

    main()
