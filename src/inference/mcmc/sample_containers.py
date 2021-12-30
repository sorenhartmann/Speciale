from typing import Callable, Dict, Iterable

from torch import Tensor


class SampleContainer:

    samples: Dict[int, Tensor]

    def register_sample(self, retrieve_sample_func: Callable[[], Tensor]) -> None:
        raise NotImplementedError

    def items(self) -> Iterable[tuple[int, Tensor]]:
        return self.samples.items()

    def values(self) -> Iterable[Tensor]:
        return self.samples.values()

    def keys(self) -> Iterable[int]:
        return self.samples.keys()

    def __iter__(self) -> Iterable[int]:
        return iter(self.samples)

    def __len__(self) -> int:
        return len(self.samples)


class FIFOSampleContainer(SampleContainer):
    """Retain as set of samples given an stream of samples of unkown length"""

    def __init__(self, max_items: int = 20, keep_every: int = 20) -> None:

        self.max_items = max_items
        self.keep_every = keep_every

        self.samples = {}
        self.stream_position = 0

    def register_sample(self, retrieve_sample_func: Callable[[], Tensor]) -> None:

        if not self.stream_position % self.keep_every == 0:
            self.stream_position += 1
            return

        if len(self.samples) == self.max_items:
            del self.samples[min(self.samples)]

        self.samples[self.stream_position] = retrieve_sample_func()
        self.stream_position += 1


class CompleteSampleContainer(SampleContainer):
    """Retain all samples"""

    def __init__(self) -> None:
        self.samples: Dict[int, Tensor] = {}
        self.stream_position = 0

    def register_sample(self, retrieve_sample_func: Callable[[], Tensor]) -> None:

        self.samples[self.stream_position] = retrieve_sample_func()
        self.stream_position += 1


class DoublingSampleContainer(SampleContainer):
    def __init__(self, max_items: int) -> None:

        self.i = 0
        self.max_items = max_items
        self.keep_indices = set(range(max_items))
        self.samples = {}

    def register_sample(self, retrieve_sample_func: Callable[[], Tensor]) -> None:

        if max(self.keep_indices) == self.i - 1:
            self.keep_indices = {(i + 1) * 2 - 1 for i in self.keep_indices}

        if len(self.samples) < self.max_items:
            pass
        elif self.i - 1 in self.keep_indices:
            remove_idx = min(set(self.samples) - self.keep_indices)
            del self.samples[remove_idx]
        else:
            del self.samples[self.i - 1]

        self.samples[self.i] = retrieve_sample_func()
        self.i += 1
