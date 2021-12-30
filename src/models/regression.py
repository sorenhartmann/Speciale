from typing import Dict

import torch
import torch.nn as nn
import torchmetrics
from torch import Tensor

from src.models.base import Model


class RegressionModel(Model):
    def __init__(self, n_vars):

        super().__init__()

        self.linear = nn.Linear(n_vars, 1)

    def forward(self, x: Tensor):
        return self.linear(x)

    def observation_model_gvn_output(self, output: Tensor):
        """Returns p(y | x, theta) given the model output, f(x)"""
        return torch.distributions.Normal(output, 1.0)

    def get_metrics(self) -> Dict[str, torchmetrics.Metric]:
        return {"mse": torchmetrics.MeanSquaredError()}
