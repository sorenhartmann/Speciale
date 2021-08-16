from abc import ABC

import tqdm
import sys

class Callback:
    pass

class ProgressBar(Callback):

    def on_burn_in_start(self, inference):
        self._bar = tqdm.tqdm(total=inference.burn_in)

    def on_burn_in_end(self, inference):
        self._bar.close()

    def on_sample_accepted(self, inference):
        self._bar.update(1)

    def on_sampling_start(self, inference):
        self._bar = tqdm.tqdm(total=inference.n_samples)

    def on_sampling_end(self, inference):
        self._bar.close()

class ProgressBarWithAcceptanceRatio(Callback):
    def _set_postfix(self):
        self._bar.set_postfix(
            {"acc_ratio": self._accepted / self._total}, refresh=False
        )

    def on_sample_accepted(self, inference):

        self._accepted += 1
        self._total += 1
        self._bar.update()
        self._set_postfix()

    def on_sample_rejected(self, inference):
        self._total += 1
        self._set_postfix()

    def on_burn_in_start(self, inference):
        self._bar = tqdm.tqdm(total=inference.burn_in)
        self._accepted = 0
        self._total = 0

    def on_burn_in_end(self, inference):
        self._bar.close()

    def on_sampling_start(self, inference):
        self._bar = tqdm.tqdm(total=inference.n_samples)
        self._accepted = 0
        self._total = 0

    def on_sampling_end(self, inference):
        self._bar.close()


class CallbackHookMixin(ABC):
    def callback(self, callback_name: str):
        for callback in self.callbacks:
            if hasattr(callback, callback_name):
                getattr(callback, callback_name)(self)
