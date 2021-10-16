
from abc import ABC, abstractmethod, abstractproperty
from contextlib import contextmanager

import torch

from src.inference.probabilistic import ProbabilisticModel
from src.utils import ParameterView_


class Samplable(ABC):

    @abstractproperty
    def state(self) -> torch.Tensor:
        pass

    @abstractmethod
    def prop_log_p(self) -> torch.Tensor:
        pass

    @abstractmethod
    def grad_prop_log_p(self) -> torch.Tensor:
        pass

    @property
    def shape(self):
        return self.state.shape


class ParameterPosterior(Samplable):
    
    """Posterior of model parameters given observations"""

    def __init__(self, model : ProbabilisticModel):

        super().__init__()

        self.model = model
        self.view = ParameterView_(model)

        self._x = None
        self._y = None
        self._sampling_fraction = 1.0

    def prop_log_p(self) -> torch.Tensor:
        return (
            self.model.log_prior()
            + self.model.log_likelihood(x=self._x, y=self._y).sum()
            / self._sampling_fraction
        )

    def grad_prop_log_p(self):
        self.model.zero_grad()
        t = self.prop_log_p()
        t.backward()
        return self.view.flat_grad

    def set_observation(self, x=None, y=None, sampling_fraction: float = 1.0):

        self._x = x
        self._y = y
        self._sampling_fraction = sampling_fraction

    @contextmanager
    def observe(self, x=None, y=None, sampling_fraction: float = 1.0):

        x_prev = self._x
        y_prev = self._y
        sampling_fraction_prev = self._sampling_fraction

        try:
            self.set_observation(x, y, sampling_fraction)
            yield
        finally:
            self.set_observation(x_prev, y_prev, sampling_fraction_prev)

    @property
    def state(self):
        return self.view[:]

    @property
    def shape(self):
        return (self.view.n_params,)

    @state.setter
    def state(self, value):
        self.model.load_state_dict(
            {
                k: value[a:b].view(shape)
                for (k, shape), (a, b) in zip(
                    self.view.param_shapes.items(), self.view.flat_index_pairs
                )
            },
            strict=False,
        )
