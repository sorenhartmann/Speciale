import torch.optim
from torch import nn
from src.models.base import Model

from .base import InferenceModule
from .probabilistic import as_probabilistic_model

from copy import deepcopy

class SGDInference(InferenceModule):

    def __init__(
        self,
        model: Model,
        lr: float = 1e-3,
        use_map=False,
        prior_spec=None
    ):

        super().__init__()

        self.model = as_probabilistic_model(model, prior_spec)
        self.lr = lr
        self.use_map = use_map

        self.train_metrics = nn.ModuleDict(self.model.get_metrics())
        self.val_metrics = nn.ModuleDict(self.model.get_metrics())

    def training_step(self, batch, batch_idx):

        x, y = batch
        output = self.model(x)

        N = len(self.trainer.train_dataloader.dataset)

        loss = self.model.loss(output, y)
        if self.use_map:
            loss -= self.model.log_prior() / N

        self.log("loss/train", loss)
        for name, metric in self.train_metrics.items():
            self.log(f"{name}/train", metric(output, y))

        return loss

    def validation_step(self, batch, batch_idx):

        x, y = batch
        output = self.model(x)

        for name, metric in self.val_metrics.items():
            self.log(f"{name}/val", metric(output, y), prog_bar=True)

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer