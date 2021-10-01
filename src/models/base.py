import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from typing import Dict
from torch import Tensor

class Model(nn.Module):

    def observation_model_gvn_output(self, output: Tensor):
        """Returns p(y | x, theta) given the model output, f(x)"""
        raise NotImplementedError

    def observation_model(self, x: Tensor):
        """Returns p(y | x, theta) given observation x"""
        return self.observation_model_gvn_output(self.forward(x))

    def loss(self, output: Tensor, targets: Tensor):
        """General loss implementation given model output"""
        return -self.observation_model_gvn_output(output).log_prob(y).mean()

    def get_metrics(self) -> Dict[str, torchmetrics.Metric]:
        """Metrics relevant for model"""
        return {}

class ErrorRate(torchmetrics.Accuracy):

    def compute(self) -> Tensor:
        return 1 - super().compute()

class ClassifierMixin:

    def observation_model_gvn_output(self, logits: torch.FloatTensor):
        return torch.distributions.Categorical(logits=logits)

    def loss(self, output: torch.FloatTensor, y: torch.FloatTensor):
        return F.cross_entropy(output, y)
    
    def get_metrics(self):        
        return {"err" : ErrorRate()}
