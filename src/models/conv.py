
import torch
import torch.nn as nn

from src.utils import SequentialBuilder

from .base import ClassifierMixin, Model


class ConvModel(Model):
    
    def __init__(self, in_shape=(3, 32, 32), out_features=10):

        super().__init__()

        seq_builder = SequentialBuilder(in_shape=in_shape)

        add = seq_builder.add
        out_dim = seq_builder.out_dim

        add(nn.Conv2d(out_dim(0), 6, 5))
        add(nn.ReLU(inplace=True))
        add(nn.MaxPool2d(2, 2))
        add(nn.Conv2d(out_dim(0), 16, 5))
        add(nn.ReLU(inplace=True))
        add(nn.MaxPool2d(2, 2))
        add(torch.nn.Flatten())
        add(nn.Linear(out_dim(0), 120))
        add(nn.ReLU(inplace=True))
        add(nn.Linear(out_dim(0), 84))
        add(nn.ReLU(inplace=True))
        add(nn.Linear(out_dim(0), out_features))

        self.net = seq_builder.build()

    def forward(self, x: torch.Tensor):
        return self.net(x)


class ConvClassifier(ClassifierMixin, ConvModel):
    ...
