from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Generator, Optional

from torch import Size, Tensor

from src.bayesian.core import log_likelihood, log_prior
from src.models.base import Model
from src.utils import ParameterView


class Samplable(ABC):
    @abstractmethod
    def get_state(self) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def set_state(self, new_state: Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def prop_log_p(self) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def grad_prop_log_p(self) -> Tensor:
        raise NotImplementedError

    @property
    def shape(self) -> Size:
        return self.get_state().shape

    @property
    def state(self) -> Tensor:
        return self.get_state()

    @state.setter
    def state(self, new_state: Tensor) -> None:
        self.set_state(new_state)


class ParameterPosterior(Samplable):

    """Posterior of model parameters given observations"""

    def __init__(self, model: Model, temperature: float = 1.0) -> None:

        super().__init__()

        self.model = model
        self.view = ParameterView(model)
        self.temperature = temperature

        self._x: Optional[Tensor] = None
        self._y: Optional[Tensor] = None
        self._sampling_fraction = 1.0

    def prop_log_p(self) -> Tensor:
        prop_log_p = (
            log_prior(self.model) + self.log_likelihood() / self._sampling_fraction
        )
        # TODO: Add temperature?
        return prop_log_p

    def log_likelihood(self) -> Tensor:
        assert self._x is not None and self._y is not None
        return log_likelihood(self.model, x=self._x, y=self._y).sum()

    def grad_prop_log_p(self) -> Tensor:
        self.model.zero_grad()
        t = self.prop_log_p()
        t.backward()
        return self.view.flat_grad

    def set_observation(
        self,
        x: Optional[Tensor] = None,
        y: Optional[Tensor] = None,
        sampling_fraction: float = 1.0,
    ) -> None:

        self._x = x
        self._y = y
        self._sampling_fraction = sampling_fraction

    @contextmanager
    def observe(
        self,
        x: Optional[Tensor] = None,
        y: Optional[Tensor] = None,
        sampling_fraction: float = 1.0,
    ) -> Generator[None, None, None]:

        x_prev = self._x
        y_prev = self._y
        sampling_fraction_prev = self._sampling_fraction

        try:
            self.set_observation(x, y, sampling_fraction)
            yield
        finally:
            self.set_observation(x_prev, y_prev, sampling_fraction_prev)

    @property
    def shape(self) -> Size:
        return Size((self.view.n_params,))

    def get_state(self) -> Tensor:
        return self.view[:]

    def set_state(self, new_state: Tensor) -> None:
        self.model.load_state_dict(
            {
                k: new_state[a:b].view(shape)
                for (k, shape), (a, b) in zip(
                    self.view.param_shapes.items(), self.view.flat_index_pairs
                )
            },
            strict=False,
        )
