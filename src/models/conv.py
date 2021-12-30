import torch
import torch.nn as nn

from src.utils import SequentialBuilder

from .base import ClassifierMixin, Model


class ConvModel(Model):
    def __init__(self, in_shape=(3, 32, 32), out_features=10, dropout=0.0):

        super().__init__()

        seq_builder = SequentialBuilder(in_shape=in_shape)

        add = seq_builder.add
        out_dim = seq_builder.out_dim

        add(nn.Conv2d(out_dim(0), 40, 5, 1, 2))
        add(nn.BatchNorm2d(out_dim(0)))
        add(nn.ReLU())
        add(nn.MaxPool2d(2, 2))

        add(nn.Conv2d(out_dim(0), 40, 5, 1, 2))
        add(nn.BatchNorm2d(out_dim(0)))
        add(nn.ReLU())
        add(nn.MaxPool2d(2, 2))

        add(nn.Conv2d(out_dim(0), 40, 5, 1, 2))
        add(nn.BatchNorm2d(out_dim(0)))
        add(nn.ReLU())
        add(nn.MaxPool2d(2, 2))

        add(torch.nn.Flatten())
        add(nn.Linear(out_dim(0), 256))
        add(nn.ReLU())
        if dropout > 0.0:
            add(nn.Dropout(dropout))
        add(nn.Linear(out_dim(0), out_features))

        self.net = seq_builder.build()

    def forward(self, x: torch.Tensor):
        return self.net(x)


class ConvClassifier(ClassifierMixin, ConvModel):
    ...
