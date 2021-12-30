import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Categorical, MixtureSameFamily, Normal


class Prior(nn.Module):
    def log_prob(self, parameter: Tensor) -> Tensor:
        raise NotImplementedError


class NormalPrior(Prior):
    def __init__(self, precision: float = 1.0, mean: float = 0.0) -> None:

        super().__init__()

        self.register_buffer("precision", torch.tensor(precision))
        self.precision: Tensor
        self.register_buffer("mean", torch.tensor(mean))
        self.mean: Tensor

    def log_prob(self, parameter: Tensor) -> Tensor:
        return Normal(self.mean, 1.0 / self.precision.sqrt()).log_prob(parameter)


class ScaleMixturePrior(Prior):
    def __init__(
        self,
        mean_1: float = 0.0,
        mean_2: float = 0.0,
        log_sigma_1: float = -1,
        log_sigma_2: float = -7,
        mixture_ratio: float = 0.5,
    ):

        super().__init__()

        self.register_buffer("mean", torch.tensor([mean_1, mean_2]))
        self.mean: Tensor
        self.register_buffer("log_sigma", torch.tensor([log_sigma_1, log_sigma_2]))
        self.log_sigma: Tensor
        mixture_logits = torch.tensor([mixture_ratio, 1 - mixture_ratio])
        self.register_buffer("mixture_logits", mixture_logits)
        self.mixture_logits: Tensor

    def log_prob(self, parameter: Tensor) -> Tensor:
        dist = MixtureSameFamily(
            Categorical(self.mixture_logits),
            Normal(self.mean, self.log_sigma.exp()),
        )
        return dist.log_prob(parameter)
