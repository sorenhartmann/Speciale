import functools
from math import exp, pi

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import iv
from src.samplers import Hamiltonian, Samplable, StochasticGradientHamiltonian


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
    xx = torch.linspace(-3, 3, 200)

    plt.plot(xx, p_exact(xx), label="True distribution")

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
    
    def experiment(sampler, samplable, *args, **kwargs):
        samples = torch.empty((n_samples,))
        sampler.setup(samplable)
        for i in range(n_samples):
            samples[i] = sampler.next_sample()
        plot_dist(samples, bins, *args, **kwargs)

    experiment(hmc, wo_noise, "-v", fillstyle="none", label="HMC")
    print("Experiment done")
    experiment(hmc_wo_acc_step, wo_noise, "-.", label="HMC (No MH)")
    print("Experiment done")
    experiment(hmc, w_noise, "-x", label="HMC /w noise")
    print("Experiment done")
    experiment(hmc_wo_acc_step, w_noise, "-.", label="HMC /w noise (No MH)")
    print("Experiment done")
    experiment(sghmc, w_noise, "-", label="SGHMC")
    print("Experiment done")

    plt.legend()
    plt.show()


if __name__ == "__main__":

    main()
