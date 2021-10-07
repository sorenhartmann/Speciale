from src.utils import register_component, HPARAM

@register_component("fifo")
class FIFOSampleContainer:
    """Retain as set of samples given an stream of samples of unkown length"""

    max_items : HPARAM[int]
    keep_every : HPARAM[int]

    def __init__(self, max_items=20, keep_every=20):

        self.max_items = max_items
        self.keep_every = keep_every

        self.samples = {}
        self.stream_position = 0

    def append(self, value):

        if not self.can_use_next():
            self.stream_position += 1
            return

        if len(self.samples) == self.max_items:
            del self.samples[min(self.samples)]

        self.samples[self.stream_position] = value
        self.stream_position += 1

    def can_use_next(self) -> bool:
        return self.stream_position % self.keep_every == 0

    def items(self):
        return self.samples.items()

    def values(self):
        return self.samples.values()

    def __iter__(self):
        return iter(self.samples)

    def __len__(self):
        return len(self.samples)
