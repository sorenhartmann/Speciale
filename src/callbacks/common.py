from typing import Any, Dict, List, Optional, Tuple
from pytorch_lightning import Callback, Trainer, LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor
import torch

import pandas as pd


class GetCalibrationCurve(Callback):
    def __init__(self, breaks: Optional[List[float]] = None) -> None:
        if breaks is None:
            breaks = [0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1]
        self.breaks = breaks

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.predictions: List[Tuple[Tensor, Tensor]] = []

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:

        _, y = batch
        preds = outputs["predictions"]
        self.predictions.append((preds, y))

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:

        target = torch.cat([x[1] for x in self.predictions])
        probs = torch.cat([x[0] for x in self.predictions])
        pred = probs.argmax(-1)
        (
            pd.DataFrame(probs.numpy())
            .assign(target=target.numpy())
            .melt(id_vars="target", value_name="prob", var_name="class_")
            .assign(
                target_is_class=lambda x: x.target == x.class_,
                interval=lambda x: pd.cut(x.prob, self.breaks),
            )
            .groupby(["interval"])  # "variable"
            .agg({"target_is_class": ["mean", "count"]})
            .droplevel(0, axis=1)
            .rename(columns={"mean": "proportion"})
            .assign(lower=self.breaks[:-1], upper=self.breaks[1:])
            .reset_index()
            .to_csv("calibration_data.csv")
        )


