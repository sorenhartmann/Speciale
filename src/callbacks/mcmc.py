from pytorch_lightning import Callback
import torch


class SaveSamples(Callback):

    def __init__(self, unflatten=False):
        self.unflatten=unflatten

    def on_fit_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
    
        if self.unflatten:
            res = {
                i: pl_module.posterior.view._unflatten(sample)
                for i, sample in pl_module.sample_container.items()
            }
        else:
            res = pl_module.sample_container.samples

        torch.save(res, "saved_samples.pt")
