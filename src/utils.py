import inspect
import typing
from functools import wraps
from typing import Generic, List, TypeVar

import numpy as np
import pandas as pd
from itertools import tee


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


T = TypeVar("T")


class HPARAM(Generic[T]):
    pass


class HyperparameterMixin:

    @classmethod
    def add_argparse_args(
        cls,
        parent_parser,
    ):
        """
        Adds w/e HPARAM typed attributes with __init__ defaults to argparser
        """
        init_params = inspect.signature(cls.__init__).parameters
        parser = parent_parser.add_argument_group(cls.__name__)
        for name, hparam_type in cls.__annotations__.items():
            if not getattr(hparam_type, "__origin__", None) is HPARAM:
                continue
            (type_,) = typing.get_args(hparam_type)
            parser.add_argument(
                f"--{name}", type=type_, default=init_params[name].default
            )

        return parent_parser

    @classmethod
    def from_argparse_args(cls, args, **kwargs):

        hparams = {
            name: getattr(args, name)
            for name, type_ in cls.__annotations__.items()
            if getattr(type_, "__origin__", None) is HPARAM
        }

        return cls(**kwargs, **hparams)

    def get_hparams(self):
        return {
            name : getattr(self, name) 
            for name, type_ in self.__class__.__annotations__.items()
            if getattr(type_, "__origin__", None) is HPARAM
        }

