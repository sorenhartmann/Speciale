
import torch
import torch.nn as nn

from src.utils import SequentialBuilder

from .base import ClassifierMixin, Model


class MLPModel(Model):


    def __init__(
        self,
        in_features=784,
        out_features=10,
        hidden_layers=[100],
    ):

        super().__init__()

        self.hidden_layers = hidden_layers

        seq_builder = SequentialBuilder(in_shape=(in_features,))
        for hidden_size in hidden_layers:
            seq_builder.add(nn.Linear(seq_builder.out_dim(0), hidden_size))
            seq_builder.add(nn.Sigmoid())
        seq_builder.add(nn.Linear(seq_builder.out_dim(0), out_features))

        self.ffnn = seq_builder.build()

    def forward(self, x: torch.Tensor):

        x = x.flatten(-2, -1)
        return self.ffnn(x)

class MLPClassifier(ClassifierMixin, MLPModel):
    ...