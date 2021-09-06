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



class StateSpaceTracer:

    state: List
    state_time: List
    momentum: List
    momentum_time: List

    def reset(self):

        self.state = []
        self.momentum = []
        self.state_time = []
        self.momentum_time = []

    def attach(self, sampler):
        
        def log_momentum(function):
            @wraps(function)
            def wrapper(*args, **kwargs):

                if len(self.momentum) == 0:
                    self.momentum = [sampler.momentum.clone()]
                    self.momentum_time = [0]

                dt = sampler.step_size
                if "half_step" in kwargs and kwargs["half_step"]:
                    dt /= 2
                function(*args, **kwargs)

                self.momentum_time.append(self.momentum_time[-1] + dt)
                self.momentum.append(sampler.momentum.clone())

            return wrapper

        def log_state(function):
            @wraps(function)
            def wrapper(*args, **kwargs):

                if len(self.state) == 0:
                    self.state = [sampler.samplable.state.clone()]
                    self.state_time = [0]

                dt = sampler.step_size
                function(*args, **kwargs)
                self.state_time.append(self.state_time[-1] + dt)
                self.state.append(sampler.samplable.state.clone())

            return wrapper

        sampler.step_momentum = log_momentum(sampler.step_momentum)
        sampler.step_parameters = log_state(sampler.step_parameters)

        self.reset()

        return self

    def df(self):

        result =  pd.merge_ordered(
            pd.DataFrame(
                {
                    "time": np.array(self.momentum_time),
                    "momentum": np.array(self.momentum),
                },
            ),
            pd.DataFrame(
                {
                    "time": np.array(self.state_time),
                    "value": np.array(self.state),
                },
            ),
            on="time",
        )

        idx = result.index[~result["value"].isna()]
        return result.interpolate().loc[idx]


def get_traces(sampler : "Sampler", n=10): #  noqa

    tracer = StateSpaceTracer().attach(sampler)
    trace_data = pd.DataFrame(columns=["trace", "time", "value", "momentum"])
    prev_sample = sampler.samplable.state.clone()

    for i in range(n):
        
        sample = sampler.next_sample()
        single_trace_data = tracer.df()
        single_trace_data["trace"] = i
        single_trace_data["accepted"] = (sample != prev_sample).item()
        trace_data = trace_data.append(single_trace_data)
        
        prev_sample = sample
        tracer.reset()

    trace_data.set_index(["trace", "time"], inplace=True)
    return trace_data
