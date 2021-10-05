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

    def observation_model(self, input: Tensor):
        """Returns p(y | x, theta) given observation x"""
        return self.observation_model_gvn_output(self.forward(input))

    def loss(self, output: Tensor, target: Tensor):
        """General loss implementation given model output"""
        return -self.observation_model_gvn_output(output).log_prob(target).mean()

    def get_metrics(self) -> Dict[str, torchmetrics.Metric]:
        """Metrics relevant for model"""
        return {}

class ErrorRate(torchmetrics.Accuracy):

    def compute(self) -> Tensor:
        return 1 - super().compute()

class ClassifierMixin:

    def observation_model_gvn_output(self, logits: torch.FloatTensor):
        return torch.distributions.Categorical(logits=logits)

    def loss(self, output: torch.FloatTensor, target: torch.FloatTensor):
        return F.cross_entropy(output, target)
    
    def predict(self, input):
        return self.forward(input).softmax(-1)

    def get_metrics(self):        
        return {"err" : ErrorRate()}
