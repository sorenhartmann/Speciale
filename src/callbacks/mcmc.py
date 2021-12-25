from functools import cache
from pathlib import Path
from typing import Any, Optional, Union
from pytorch_lightning import Callback, callbacks
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from src.inference.mcmc.variance_estimators import WelfordEstimator
from src.inference.mcmc.samplers import SGHMCWithVarianceEstimator
from src.models.base import ErrorRate
import pandas as pd
import torchmetrics.functional as FM


class SaveSamples(Callback):
    def __init__(self, unflatten=False):
        self.unflatten = unflatten

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


class SGHMCLogGradientVariance(Callback):
    def __init__(
        self,
        n_gradients: int = 1000,
        steps_per_log: int = 50,
        estimation_steps: int = 100,
        path: Union[str, Path] = "variance_estimates.pt",
    ) -> None:
        super().__init__()
        self.n_gradients = n_gradients
        self.steps_per_log = steps_per_log
        self.estimation_steps = estimation_steps
        self.path = Path(path)

    def on_fit_start(self, trainer, pl_module) -> None:

        with torch.random.fork_rng():
            torch.manual_seed(123)
            self.log_idx, _ = torch.sort(
                torch.randperm(pl_module.posterior.shape[0])[: self.n_gradients]
            )
            self.estimates = {}

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, unused
    ) -> None:

        if trainer.global_step % self.steps_per_log == 0:
            self.log_gradient_estimate(trainer, pl_module, batch_idx)

    def log_gradient_estimate(self, trainer, pl_module, batch_idx):

        with torch.random.fork_rng():

            wf_estimator = WelfordEstimator(pl_module.posterior.shape)
            for _ in range(self.estimation_steps):
                x, y = next(iter(trainer.train_dataloader))
                sampling_fraction = len(x) / len(trainer.train_dataloader.dataset)
                with pl_module.posterior.observe(x, y, sampling_fraction):
                    wf_estimator.update(pl_module.posterior.grad_prop_log_p())

        observed_variance = wf_estimator.estimate()
        estimated_variance = pl_module.sampler.variance_estimator.estimate()

        observed_variance, estimated_variance = torch.broadcast_tensors(
            observed_variance, estimated_variance
        )

        self.estimates[trainer.global_step] = {
            "observed_variance": observed_variance,
            "estimated_variance": estimated_variance,
        }

    def on_fit_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        torch.save({"log_idx": self.log_idx, "estimates": self.estimates}, self.path)


class SGHMCLogTemperature(Callback):
    def __init__(
        self, steps_per_log: int = 50, path: Union[str, Path] = "temperature_samples.pt"
    ):

        self.path = path
        self.temperature_samples = {}
        self.steps_per_log = steps_per_log

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:

        if trainer.global_step % self.steps_per_log != 0:
            return

        if isinstance(pl_module.sampler, SGHMCWithVarianceEstimator):
            M_diag = pl_module.sampler.mass_factor
            step_size = torch.sqrt(pl_module.sampler.lr_0)
            nu = pl_module.sampler.nu
            r = nu / step_size * M_diag
            norm_squares = (1 / M_diag) * r * r
        else:
            step_size = torch.sqrt(pl_module.sampler.lr)
            nu = pl_module.sampler.nu
            r = nu / step_size
            norm_squares = r * r

        unflattened = pl_module.posterior.view._unflatten(norm_squares)
        for k, v in unflattened.items():
            self.temperature_samples[trainer.global_step, k] = {
                "temperature_sum": v.sum().item(),
                "n_params": v.numel(),
            }

    def on_fit_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        torch.save(self.temperature_samples, self.path)


# class LogSampleLikelihood(Callback):
#     def on_fit_end(
#         self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
#     ) -> None:
#         torch.save(pl_module.val_joint_logliks, "val_joint_logliks.pt")
#         torch.save(pl_module.train_joint_logliks, "train_joint_logliks.pt")
#         torch.save(pl_module.val_avg_likelihood, "val_avg_likelihood.pt")


# class SaveTestPredictions(Callback):

#     def on_test_batch_end(
#         self,
#         trainer: "pl.Trainer",
#         pl_module: "pl.LightningModule",
#         outputs: Optional[STEP_OUTPUT],
#         batch: Any,
#         batch_idx: int,
#         dataloader_idx: int,
#     ) -> None:

#         pred = 0
#         for i, sample_idx in enumerate(self.samples_order):
#             pred += outputs["predictions"][sample_idx]
#         pred.soft_max(-1)


class GetSampleFilterCurve(Callback):
    def on_test_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:

        sample_idx_order = torch.randperm(
            len(pl_module.sample_container), generator=torch.Generator().manual_seed(24)
        ).tolist()
        sample_idx =  list(pl_module.sample_container)
        self.samples_order = [sample_idx[i] for i in sample_idx_order]
        self.results = []

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:

        _, y = batch
        pred = 0
        for i, sample_idx in enumerate(self.samples_order):
            pred += outputs["per_sample_predictions"][sample_idx]
            error_rate = 1 - FM.accuracy(pred, y)
            self.results.append(
                {
                    "batch_idx": batch_idx,
                    "with_n_samples": i + 1,
                    "error_rate": error_rate.item(),
                }
            )

    def on_test_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        stats: pd.DataFrame = (
            pd.DataFrame(self.results)
            .set_index("batch_idx")
            .groupby("with_n_samples")
            .mean("error_rate")
        )
        stats.to_json("sample_resampling_curve.json")
