from typing import Dict, Any

class SampleContainer:

    samples: Dict[int, Any]

    def register_sample(self, retrieve_sample_func):
        raise NotImplementedError

    def items(self):
        return self.samples.items()

    def values(self):
        return self.samples.values()

    def __iter__(self):
        return iter(self.samples)

    def __len__(self):
        return len(self.samples)


class FIFOSampleContainer(SampleContainer):
    """Retain as set of samples given an stream of samples of unkown length"""

    def __init__(self, max_items=20, keep_every=20):

        self.max_items = max_items
        self.keep_every = keep_every

        self.samples = {}
        self.stream_position = 0

    def register_sample(self, retrieve_sample_func):

        if not self.stream_position % self.keep_every == 0:
            self.stream_position += 1
            return

        if len(self.samples) == self.max_items:
            del self.samples[min(self.samples)]

        self.samples[self.stream_position] = retrieve_sample_func()
        self.stream_position += 1

class CompleteSampleContainer(SampleContainer):
    """Retain all samples"""

    def __init__(self):
        self.samples = {}
        self.stream_position = 0

    def register_sample(self, retrieve_sample_func):

        self.samples[self.stream_position] = retrieve_sample_func()
        self.stream_position += 1

class DoublingSampleContainer(SampleContainer):

    def __init__(self, max_items):
        
        self.i = 0
        self.max_items = max_items
        self.keep_indices = set(range(max_items))
        self.samples = {}

    def register_sample(self, retrieve_sample_func):

        if max(self.keep_indices) == self.i - 1:
            self.keep_indices = {(i+1)*2-1 for i in self.keep_indices}
        
        if len(self.samples) < self.max_items:
            pass
        elif self.i-1 in self.keep_indices:
            remove_idx = min(set(self.samples) - self.keep_indices)
            del self.samples[remove_idx]
        else:
            del self.samples[self.i-1]

        self.samples[self.i] = retrieve_sample_func()
        self.i += 1