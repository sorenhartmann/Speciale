from src.inference.mcmc.samplable import Samplable
import torch
import numpy as np

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

