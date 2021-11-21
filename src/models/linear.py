from typing import Dict, List

import torch.nn as nn
from torch import Tensor

from src.models.base import ClassifierMixin, Model
import torch

class LinearClassifier(ClassifierMixin, Model):

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features-1, bias)

    def forward(self, x):
        x = self.linear.forward(x)
        x = torch.cat([x, torch.zeros((*x.shape[:-1], 1))], dim=-1)
        return x