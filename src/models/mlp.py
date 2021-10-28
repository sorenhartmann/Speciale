from typing import Optional
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
        activation_func: Optional[nn.Module] = None,
        dropout: float = None,
    ):

        super().__init__()

        if activation_func is None:
            activation_func = nn.Sigmoid()

        self.hidden_layers = hidden_layers

        if dropout is not None:
            dropout_module = nn.Dropout(dropout)

        seq_builder = SequentialBuilder(in_shape=(in_features,))
        for hidden_size in hidden_layers:
            seq_builder.add(nn.Linear(seq_builder.out_dim(0), hidden_size))
            seq_builder.add(activation_func)
            if dropout is not None:
                seq_builder.add(dropout_module)

        seq_builder.add(nn.Linear(seq_builder.out_dim(0), out_features))

        self.ffnn = seq_builder.build()

    def forward(self, x: torch.Tensor):

        x = x.flatten(-2, -1)
        return self.ffnn(x)


class MLPClassifier(ClassifierMixin, MLPModel):
    ...
