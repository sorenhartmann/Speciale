import torch.optim
from torch import nn
from src.models.base import ErrorRate, Model

from .base import InferenceModule
from src.bayesian.core import to_bayesian_model, log_prior

from copy import deepcopy

class SGDInference(InferenceModule):

    def __init__(
        self,
        model: Model,
        lr: float = 1e-3,
        use_map=False,
        prior_config=None
    ):

        super().__init__()

        self.lr = lr
        self.use_map = use_map
        if self.use_map:
            model = to_bayesian_model(model, prior_config)

        self.model = model

        self.train_metrics = nn.ModuleDict(self.model.get_metrics())
        self.val_metrics = nn.ModuleDict(self.model.get_metrics())

    def training_step(self, batch, batch_idx):

        x, y = batch
        output = self.model(x)

        N = len(self.trainer.train_dataloader.dataset)

        loss = self.model.loss(output, y)
        if self.use_map:
            loss -= log_prior(self.model) / N

        self.log("loss/train", loss)
        for name, metric in self.train_metrics.items():
            self.log(f"{name}/train", metric(output, y))

        return loss

    def validation_step(self, batch, batch_idx):

        x, y = batch
        output = self.model(x)

        for name, metric in self.val_metrics.items():
            self.log(f"{name}/val", metric(output, y), prog_bar=True)

    def on_test_epoch_start(self) -> None:
        self.test_metric = ErrorRate().to(device=self.device)

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)
        self.log(f"err/test", self.test_metric(output, y), prog_bar=True)
        return {"predictions": output.softmax(-1)}

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
