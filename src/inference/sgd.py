from typing import Optional, Sized, cast

import torch.optim
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn

from src.bayesian.core import BayesianConversionConfig, log_prior, to_bayesian_model
from src.models.base import ErrorRate, Model

from .base import BATCH_IN, InferenceModule


class SGDInference(InferenceModule):
    def __init__(
        self,
        model: Model,
        lr: float = 1e-3,
        use_map: bool = False,
        prior_config: Optional[BayesianConversionConfig] = None,
    ):

        super().__init__()

        self.lr = lr
        self.use_map = use_map

        if self.use_map:
            model = to_bayesian_model(model, prior_config)

        self.model = model

        self.train_metrics = nn.ModuleDict(self.model.get_metrics())
        self.val_metrics = nn.ModuleDict(self.model.get_metrics())

    def training_step(self, batch: BATCH_IN, batch_idx: int) -> Optional[STEP_OUTPUT]:  # type: ignore

        x, y = batch

        trainer = cast(Trainer, self.trainer)

        output = self.model(x)

        N = len(cast(Sized, trainer.train_dataloader.dataset))

        loss = self.model.loss(output, y)
        if self.use_map:
            loss -= log_prior(self.model) / N

        self.log("loss/train", loss)
        for name, metric in self.train_metrics.items():
            self.log(f"{name}/train", metric(output, y))

        return loss

    def validation_step(self, batch: BATCH_IN, batch_idx: int) -> Optional[STEP_OUTPUT]:  # type: ignore

        x, y = batch
        output = self.model(x)

        for name, metric in self.val_metrics.items():
            self.log(f"{name}/val", metric(output, y), prog_bar=True)

    def on_test_epoch_start(self) -> None:
        self.test_metric = ErrorRate().to(device=self.device)

    def test_step(self, batch: BATCH_IN, batch_idx: int) -> Optional[STEP_OUTPUT]:  # type: ignore

        x, y = batch
        output = self.model(x)
        self.log(f"err/test", self.test_metric(output, y), prog_bar=True)
        return {"predictions": output.softmax(-1)}

    def configure_optimizers(self) -> torch.optim.Optimizer:

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
