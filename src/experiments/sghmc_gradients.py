from pathlib import Path

import torch
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from src.data.mnist import MNISTDataModule
from src.experiments.common import ExperimentHandler, FlatTensorBoardLogger
from src.inference import BayesianClassifier


class LogGradientVariance(Callback):

    def __init__(self):
        self.gradient_vars = []

    def on_train_epoch_end(self, trainer: Trainer, pl_module) -> None:

        gradients = []
        with torch.random.fork_rng():
            for batch in trainer.train_dataloader:
                x, y = batch
                sampling_fraction = len(x) / len(trainer.train_dataloader.dataset)
                with pl_module.posterior.observe(x, y, sampling_fraction):
                    gradients.append(pl_module.posterior.grad_prop_log_p())

            variance = torch.var(torch.stack(gradients), 0, unbiased=True)
            pl_module.log("avg_grad_variance", variance.mean().item())

        self.gradient_vars.append(variance)

        if trainer.current_epoch % 50 == 0:

            variance_images = pl_module.posterior.unflatten(variance)
            for name, img in variance_images.items():
                shape = img.shape
                shape = (1,) * (3 - len(shape)) + shape
                trainer.logger.experiment.add_image(
                    f"gradient_variance/{name}",
                    img.view(*shape),
                    global_step=trainer.current_epoch,
                )

            trainer.logger.experiment.add_histogram(
                "gradient_variance", variance, global_step=trainer.current_epoch
            )

    def on_fit_end(self, trainer, pl_module) -> None:
        torch.save(self.gradient_vars, "./gradient_vars.pkl")

class LogGradients(Callback):

    def __init__(self, logged_fraction=1, log_runs=1):

        self.logged_fraction = logged_fraction
        self.log_runs = log_runs
        self.save_dir = Path("./logged_gradients")
        self.save_dir.mkdir()

    def on_train_start(self, trainer, pl_module) -> None:
        
        interval_length = trainer.max_epochs * self.logged_fraction / self.log_runs

        gap = (trainer.max_epochs - trainer.max_epochs * self.logged_fraction) / (
            max(self.log_runs - 1, 1)
        )

        start = 0
        end = interval_length

        self.intervals = []
        for _ in range(self.log_runs):
            self.intervals.append((int(start), int(end)))
            start = end + gap
            end = start + interval_length

    def on_train_epoch_start(self, trainer, pl_module) -> None:

        e = trainer.current_epoch

        if any( a <= e <= b for a, b in self.intervals):
            self.gradients = []
        else:
            self.gradients = None

    def on_batch_end(self, trainer, pl_module) -> None:

        if self.gradients is None:
            return

        self.gradients.append(pl_module.posterior.state_grad.clone())

    def on_train_epoch_end(self, trainer, pl_module) -> None:

        if self.gradients is None:
            return

        torch.save(
            torch.stack(self.gradients),
            self.save_dir / f"gradients_epoch_{trainer.current_epoch:04}.pt",
        )

        del self.gradients


def experiment():

    torch.manual_seed(123)

    dm = MNISTDataModule(500)
    model = MNISTModel()
    sampler = StochasticGradientHamiltonian()
    inference = BayesianClassifier(model, sampler)

    logger = FlatTensorBoardLogger("./metrics")
    callbacks = [
        LogGradients(),
        LogGradientVariance(),
        ModelCheckpoint("./checkpoints"),
    ]

    Trainer(logger=logger, callbacks=callbacks).fit(inference, dm)


if __name__ == "__main__":

    handler = ExperimentHandler(experiment)
    handler.run()
