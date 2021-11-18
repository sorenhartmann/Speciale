
import torch
import torch.nn as nn
from torch.distributions import Categorical, MixtureSameFamily, Normal

class Prior(nn.Module):
    def log_prob(self, parameter):
        raise NotImplementedError


class NormalPrior(Prior):
    def __init__(self, precision=1.0, mean=0.0):

        super().__init__()

        self.register_buffer("precision", torch.tensor(precision))
        self.register_buffer("mean", torch.tensor(mean))

    def log_prob(self, parameter):
        return Normal(self.mean, 1.0 / self.precision.sqrt()).log_prob(parameter)


class ScaleMixturePrior(Prior):
    def __init__(
        self,
        mean_1=0.0,
        mean_2=0.0,
        log_sigma_1=-1,
        log_sigma_2=-7,
        mixture_ratio=0.5,
    ):

        super().__init__()

        self.register_buffer("mean", torch.tensor([mean_1, mean_2]))
        self.register_buffer("log_sigma", torch.tensor([log_sigma_1, log_sigma_2]))
        mixture_logits = torch.tensor([mixture_ratio, 1 - mixture_ratio])
        self.register_buffer("mixture_logits", mixture_logits)

    def log_prob(self, parameter):
        dist = MixtureSameFamily(
            Categorical(self.mixture_logits),
            Normal(self.mean, self.log_sigma.exp()),
        )
        return dist.log_prob(parameter)