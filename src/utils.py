import inspect
import typing
from functools import wraps
from typing import Callable, Generic, List, TypeVar

import numpy as np
import pandas as pd
from itertools import tee, accumulate

import torch.nn as nn
import math
import torch


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


T = TypeVar("T")

class Component:
    pass

class HPARAM(Generic[T]):
    pass

import torch

def _get_item(value):
    if isinstance(value, torch.Tensor):
        return value.item()
    else:
        return value

# class HyperparameterMixin:

#     @classmethod
#     def add_argparse_args(
#         cls,
#         parent_parser,
#     ):
#         """
#         Adds w/e HPARAM typed attributes with __init__ defaults to argparser
#         """
        
#         init_params = inspect.signature(cls.__init__).parameters
#         parser = parent_parser.add_argument_group(cls.__name__)
#         for name, hparam_type in cls.__annotations__.items():
#             if not getattr(hparam_type, "__origin__", None) is HPARAM:
#                 continue
#             (type_,) = typing.get_args(hparam_type)
#             parser.add_argument(
#                 f"--{name}", type=type_, default=init_params[name].default
#             )

#         return parent_parser

#     @classmethod
#     def from_argparse_args(cls, args, **kwargs):

#         hparams = {
#             name: getattr(args, name)
#             for name, type_ in cls.__annotations__.items()
#             if getattr(type_, "__origin__", None) is HPARAM
#         }

#         return cls(**kwargs, **hparams)

#     def get_hparams(self):
#         return {
#             name : _get_item(getattr(self, name))
#             for name, type_ in self.__class__.__annotations__.items()
#             if getattr(type_, "__origin__", None) is HPARAM
#         }



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

    def __init__(self, model : nn.Module):

        self.model = model
        self.param_shapes = {k: x.shape for k, x in self.model.named_parameters()}
        indices = accumulate(
            self.param_shapes.values(), lambda x, y: x + math.prod(y), initial=0
        )
        self.flat_index_pairs = list(pairwise(indices))

        self.n_params = self.flat_index_pairs[-1][-1]

    def __getitem__(self, key):

        if type(key) is slice:
            return self._get_slice(key)
        else:
            raise NotImplementedError

    def __setitem__(self, key, value):
        
        if type(key) is slice:
            self._set_slice(key, value)
    
    @property
    @torch.no_grad()
    def flat_grad(self):
        return self._flatten(x.grad for x in self.model.parameters())

    def _get_slice(self, slice_):

        if slice_.start is None and slice_.stop is None and slice_.step is None:
            return self._flatten(self.model.parameters())
        else:
            raise NotImplementedError

    def _set_slice(self, slice_, value):

        if slice_.start is None and slice_.stop is None and slice_.step is None:
            state_dict = self._unflatten(value)
            for name, parameter in self.model.named_parameters():
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

    def apply(self, fnc):
        for param in self.model.parameters():
            fnc(param)
            
class ParameterView_:

    def __init__(self, model : nn.Module, parameters=None, buffers=None):

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

        yield from ((n, buffer) for n, buffer in self.model.named_buffers() if n in self.buffers)
        yield from ((n, parameter) for n, parameter in self.model.named_parameters() if n in self.parameters)
        # yield from ((n, self.model.get_buffer(n)) for n in self.buffers)

    def attributes(self):
        yield from (buffer for n, buffer in self.model.named_buffers() if n in self.buffers)
        yield from (parameter for n, parameter in self.model.named_parameters() if n in self.parameters)
        # yield fr

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


class RegisteredComponents:

    components = {}

    @classmethod
    def register_component(cls, module_cls: type, name : str):
        cls.components[name] = module_cls

from functools import wraps
def register_component(name : str):
    def decorator(module_cls):
        RegisteredComponents.components[name] = module_cls
        return module_cls

    return decorator


