from functools import wraps
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from src.experiments.common import ExperimentHandler
from src.experiments.synthetic import Example
from src.samplers import Hamiltonian, Sampler, StochasticGradientHamiltonian


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

        result = pd.merge_ordered(
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


def _get_accepted_states(trace_data):

    return (
        trace_data.groupby(level=("sampler", "trace"))
        .last()
        .loc[lambda x: x.accepted == True]
        .drop(columns="accepted")
        .reset_index()
    )


def _get_initial_states(trace_data):

    return trace_data.groupby(level=("sampler", "trace")).first()

def _get_momentum_updates(trace_data):

    return (
        trace_data.groupby(level=("sampler", "trace"))
        .agg({"value": ["last"], "momentum": ["first", "last"]})
        .assign(
            momentum_from=lambda x: x[("momentum", "last")],
            momentum_to=lambda x: x[("momentum", "first")].shift(-1),
        )
        .drop(columns="momentum")
        .droplevel(1, axis=1)
        .reset_index()
        .melt(id_vars=["sampler", "trace", "value"], value_name="momentum")
        .drop(columns="variable")
        .sort_values(["sampler", "trace"])
        .loc[lambda x: x.trace < max(x.trace)]
        .reset_index(drop=True)
    )


def _get_traces(sampler: Sampler, n=10):  #  noqa

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

def experiment(n_samples=10, step_size=0.025, grad_noise=2.0, n_steps=50):

    torch.manual_seed(10)

    samplable = Example(grad_noise=grad_noise)
    hmc = Hamiltonian(n_steps=n_steps, step_size=step_size).setup(samplable)
    hmc_traces = _get_traces(hmc, n_samples)
    hmc_traces["sampler"] = "hmc"

    samplable.grad_noise = grad_noise
    samplable.state = torch.tensor(0.0)
    sghmc = StochasticGradientHamiltonian(
        n_steps=n_steps, step_size=step_size, M=1.0, C=1.0, V=grad_noise ** 2
    ).setup(samplable)
    sghmc_traces = _get_traces(sghmc, n_samples)
    sghmc_traces["sampler"] = "sghmc"

    trace_data = pd.concat(
        x.set_index("sampler", append=True) for x in [hmc_traces, sghmc_traces]
    )
    trace_data = trace_data.reorder_levels(["sampler", "trace", "time"])

    return trace_data

def plot_traces(run, ax=None):

    if ax is None:
        ax = plt.gca()

    trace_data = run.result

    initial_states = _get_initial_states(trace_data)
    accepted_states = _get_accepted_states(trace_data)
    momentum_updates = _get_momentum_updates(trace_data)

    sns.lineplot(
        data=trace_data,
        x="value",
        y="momentum",
        hue="sampler",
        sort=False,
        units="trace",
        estimator=None,
        hue_order=["hmc", "sghmc"],
    )
    sns.scatterplot(
        data=accepted_states,
        x="value",
        y="momentum",
        color="black",
        marker="X",
        sizes=20,
    )
    sns.scatterplot(
        data=initial_states,
        x="value",
        y="momentum",
        color="green",
        marker="o",
    )
    sns.lineplot(
        x="value",
        y="momentum",
        color="grey",
        units="sampler_trace",
        estimator=None,
        data=momentum_updates.assign(
            sampler_trace=lambda x: x.sampler + x.trace.astype(str)
        ),
        linestyle="dashed",
    )

    ax.set_xlim(-2, 2)
    ax.set_ylim(-3, 3)



def main():

    handler = ExperimentHandler(experiment)
    
    latest_run = handler.latest_run()
    plot_traces(latest_run)


if __name__ == "__main__":

    main()
