from typing import Dict, List

import torch
import torch.nn as nn
import torchmetrics
from torch import Tensor

from src.models.base import Model


class PolynomialModel(Model):
    def __init__(self, coeffs: List[float]):

        super().__init__()

        coeffs = torch.tensor(coeffs)

        self.linear = nn.Linear(len(coeffs) - 1, 1)

        with torch.no_grad():
            self.linear.bias.copy_(coeffs[0])
            self.linear.weight.copy_(coeffs[1:])

    def forward(self, x: Tensor):

        x = torch.cat([x ** i for i in range(1, self.linear.in_features + 1)], dim=-1)
        return self.linear(x)

    def observation_model_gvn_output(self, output: Tensor):
        """Returns p(y | x, theta) given the model output, f(x)"""
        return torch.distributions.Normal(output, 1.0)

    def get_metrics(self) -> Dict[str, torchmetrics.Metric]:
        return {"mse" : torchmetrics.MeanSquaredError()}