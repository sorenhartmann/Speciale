import torch.optim

from src.models.base import Model

from .base import InferenceModule
from .probabilistic import as_probabilistic_model


class MAPInference(InferenceModule):


    def __init__(self, model : Model, lr: float=1e-3):

        super().__init__()

        self.model = as_probabilistic_model(model)
        self.lr = lr


        self.train_metrics = self.model.get_metrics()
        self.val_metrics = self.model.get_metrics()

    
    def training_step(self, batch, batch_idx):

        x, y = batch
        output = self.model(x)
        
        N = len(self.trainer.train_dataloader.dataset)
        
        loss = self.model.loss(output, y)
        loss -= self.model.log_prior() / N
        
        self.log("loss/train", loss)
        for name, metric in self.train_metrics.items():
            self.log(f"{name}/train", metric(output, y), on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):

        x, y = batch
        output = self.model(x)

        for name, metric in self.val_metrics.items():
            self.log(f"{name}/val", metric(output, y))

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
