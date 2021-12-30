import torch
import torch.nn as nn
from torchmetrics import MeanSquaredError

from src.models.base import ClassifierMixin, Model


class LinearClassifier(ClassifierMixin, Model):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features - 1, bias)

    def forward(self, x):
        x = self.linear.forward(x)
        x = torch.cat([x, torch.zeros((*x.shape[:-1], 1))], dim=-1)
        return x


class LinearRegressor(Model):
    # Using design matrix X

    def __init__(self, in_features, sigma=1):
        super().__init__()
        self.sigma = sigma
        self.linear = nn.Linear(in_features, 1, bias=False)

    def forward(self, x):
        x = self.linear.forward(x)
        return x

    def observation_model_gvn_output(self, output: torch.FloatTensor):
        return torch.distributions.Normal(output, scale=self.sigma)

    def predict_gvn_output(self, output):
        return output

    def get_metrics(self):
        return {"mse": MeanSquaredError()}
