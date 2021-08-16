import math
from typing import List, Optional
import pandas as pd

import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.callbacks import Callback, CallbackHookMixin, ProgressBarWithAcceptanceRatio
from src.modules import BayesianModel
from src.samplers import Sampler


class MonteCarloInference(CallbackHookMixin):
    def __init__(
        self,
        sampler: Sampler,
        burn_in: int = 500,
        n_samples: int = 1000,
        callbacks: Optional[List[Callback]] = None,
    ):

        self.sampler = sampler
        self.burn_in = burn_in
        self.n_samples = n_samples

        if callbacks is None:
            self.callbacks = self._default_callbacks()
        else:
            self.callbacks = callbacks

        self.samples_ = None

    def burn_in_loop(self, sample_generator):

        self.callback("on_burn_in_start")
        try:
            for i in range(self.burn_in):
                while next(sample_generator) is None:
                    self.callback("on_sample_rejected")
                self.callback("on_sample_accepted")

        finally:
            self.callback("on_burn_in_end")

    def sample_loop(self, sample_generator):

        self.callback("on_sampling_start")
        samples = []

        try:
            for i in range(self.n_samples):
                while (sample := next(sample_generator)) is None:
                    self.callback("on_sample_rejected")

                self.callback("on_sample_accepted")
                samples.append(sample)
        finally:
            self.callback("on_sampling_end")

        return samples

    def fit(self, model: BayesianModel, dataset: Dataset):

        sample_generator = self.sampler.sample_generator(model, dataset)

        if self.burn_in > 0:
            self.burn_in_loop(sample_generator)

        if self.n_samples > 0:
            samples = self.sample_loop(sample_generator)

        self.samples_ = samples
        self.model_ = model

        sample_generator.close()

    def predictive(self, x: Tensor):

        with torch.no_grad():
            pred_samples = []
            for sample in self.samples_:
                self.model_.load_state_dict(sample, strict=False)
                pred_samples.append(self.model_.forward(x))

        return torch.stack(pred_samples)

    @staticmethod
    def _default_callbacks():
        return [ProgressBarWithAcceptanceRatio()]

    def sample_df(self):
        dict_list = []
        for sample in self.samples_:
            dict_list.append({})
            for name, param in sample.items():
                dict_list[-1].update(
                    {f"{name}:{i}": c.item() for i, c in enumerate(param.flatten())}
                )

        return pd.DataFrame(dict_list)
