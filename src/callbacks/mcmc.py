from pytorch_lightning import Callback
import torch


class SaveSamples(Callback):
    def on_fit_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        unflattened = {
            i: pl_module.posterior.view._unflatten(sample)
            for i, sample in pl_module.sample_container.items()
        }
        torch.save(unflattened, "saved_samples.pt")
