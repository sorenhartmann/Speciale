import math
from itertools import accumulate, tee

import torch
import torch.nn as nn


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class SequentialBuilder:

    DUMMY_BATCH_SIZE = 1

    def __init__(self, in_shape):

        self.in_shape = in_shape
        self.dummy_tensor = torch.randn((self.DUMMY_BATCH_SIZE,) + in_shape)

        self.modules = []

    @property
    def out_shape(self):
        return self.dummy_tensor.shape[1:]

    @torch.no_grad()
    def add(self, module):

        self.dummy_tensor = module(self.dummy_tensor)
        self.modules.append(module)

    def out_dim(self, dim):
        return self.out_shape[dim]

    def build(self):
        return nn.Sequential(*self.modules)


class ParameterView:

    def __init__(self, model: nn.Module, parameters=None, buffers=None):

        self.model = model

        if parameters is None and buffers is None:
            parameters = [n for n, _ in self.model.named_parameters()]
            buffers = []
        elif parameters is None:
            parameters = []
        elif buffers is None:
            buffers = []

        self.parameters = set(parameters)
        self.buffers = set(buffers)

        self.param_shapes = {k: x.shape for k, x in self.named_attributes()}

        indices = accumulate(
            self.param_shapes.values(), lambda x, y: x + math.prod(y), initial=0
        )
        self.flat_index_pairs = list(pairwise(indices))

        self.n_params = self.flat_index_pairs[-1][-1]

    def named_attributes(self):

        yield from (
            (n, buffer) for n, buffer in self.model.named_buffers() if n in self.buffers
        )
        yield from (
            (n, parameter)
            for n, parameter in self.model.named_parameters()
            if n in self.parameters
        )
        # yield from ((n, self.model.get_buffer(n)) for n in self.buffers)

    def attributes(self):
        yield from (
            buffer for n, buffer in self.model.named_buffers() if n in self.buffers
        )
        yield from (
            parameter
            for n, parameter in self.model.named_parameters()
            if n in self.parameters
        )

    def __getitem__(self, key):

        if type(key) is slice:
            return self._get_slice(key)
        else:
            raise NotImplementedError

    def __setitem__(self, key, value):

        if type(key) is slice:
            self._set_slice(key, value)

    def apply_(self, func):
        for attribute in self.attributes():
            func(attribute)

    @property
    @torch.no_grad()
    def flat_grad(self):
        return self._flatten(x.grad for x in self.attributes())

    def _get_slice(self, slice_):

        if slice_.start is None and slice_.stop is None and slice_.step is None:
            return self._flatten(self.attributes())
        else:
            raise NotImplementedError

    def _set_slice(self, slice_, value):

        if slice_.start is None and slice_.stop is None and slice_.step is None:
            state_dict = self._unflatten(value)
            for name, parameter in self.named_attributes():

                #     if parameter.requires_grad:
                #         parameter.copy_(state_dict[name])
                # else:
                parameter.detach_()
                parameter.copy_(state_dict[name])
        else:
            raise NotImplementedError

    def _flatten(self, tensor_iter):
        return torch.cat([x.flatten() for x in tensor_iter])

    def _unflatten(self, value):
        return {
            k: value[a:b].view(shape)
            for (k, shape), (a, b) in zip(
                self.param_shapes.items(), self.flat_index_pairs
            )
        }


import warnings


def silence_warnings():
    warnings.filterwarnings("ignore", "`LightningModule.configure_optimizers` returned `None`")
    warnings.filterwarnings("ignore", ".+does not have many workers which may be a bottleneck.")
    warnings.filterwarnings("ignore", "The given NumPy array is not writeable,")

from pytorch_lightning import Callback


class SilenceWarnings(Callback):

    def on_init_start(self, trainer):
        silence_warnings()
