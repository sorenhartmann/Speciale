from typing import Dict, List, Optional, cast

import pandas as pd
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor
from torchmetrics import CalibrationError

from src.inference.base import BATCH_IN
from src.utils import pairwise


class GetCalibrationCurve(Callback):
    def __init__(self, n_bins: int = 10) -> None:
        self.n_bins = n_bins

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.predictions: List[Dict[str, Tensor]] = []
        self.calibration_error = CalibrationError().to(device=pl_module.device)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: BATCH_IN,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:

        assert isinstance(outputs, dict)

        _, y = batch

        self.calibration_error(outputs["predictions"], y)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:

        pl_module.log("ece/test", self.calibration_error.compute())

        confidences = cast(List[Tensor], self.calibration_error.confidences)
        accuracies = cast(List[Tensor], self.calibration_error.accuracies)

        confs = torch.cat(confidences).to(device="cpu").numpy()
        accs = torch.cat(accuracies).to(device="cpu").numpy()

        bins = cast(Tensor, self.calibration_error.bin_boundaries).tolist()
        bin_labels = [f"({a:.2}, {b:.2}]" for a, b in pairwise(bins)]

        (
            pd.DataFrame({"confidence": confs, "accuracy": accs})
            .assign(bin=lambda x: pd.cut(x.confidence, bins, labels=bin_labels))
            .groupby("bin")
            .agg(
                mean_confidence=pd.NamedAgg("confidence", aggfunc="mean"),
                mean_accuracy=pd.NamedAgg("accuracy", aggfunc="mean"),
                count=pd.NamedAgg("accuracy", aggfunc="count"),
            )
            .reset_index()
            .to_csv("ce_stats.csv", index=False)
        )
