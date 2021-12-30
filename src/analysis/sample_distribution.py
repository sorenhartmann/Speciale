from functools import cache
from typing import List, Dict
from torch import Tensor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from numpy.typing import NDArray

from src.utils import Run

@cache
def load_samples(run: Run) -> pd.DataFrame:

    def to_matrix(samples: Dict[int, Tensor]) -> NDArray:
        return torch.stack(list(samples.values())).numpy()

    return (
        pd.DataFrame(to_matrix(torch.load(run._dir / "saved_samples.pt")))
        .rename_axis(index="sample")
        .assign(
            sampler=run.cfg["inference"]["sampler"]["_target_"],
            batch_size=run.cfg["data"]["batch_size"],
        )
        .assign(
            sampler=lambda x: x.sampler.str.extract(r"src.inference.mcmc.samplers.(.+)")
        )
        .set_index(["sampler", "batch_size"], append=True)
        .reorder_levels(["sampler", "batch_size", "sample"])
    )
