from pathlib import Path
from typing import Dict, Optional, Sized, Tuple, Union, cast

import pandas as pd
import torch
import torchmetrics.functional as FM
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch.types import Number

from src.inference.base import BATCH_IN
from src.inference.mcmc import MCMCInference
from src.inference.mcmc.samplers import SGHMC, SGHMCWithVarianceEstimator
from src.inference.mcmc.variance_estimators import WelfordEstimator


class SaveSamples(Callback):
    def __init__(self, unflatten: bool = False) -> None:
        self.unflatten = unflatten

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:

        assert isinstance(pl_module, MCMCInference)

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:

        pl_module_ = cast(MCMCInference, pl_module)

        res: Union[Dict[int, Tensor], Dict[int, Dict[str, Tensor]]]

        if self.unflatten:
            res = {
                i: pl_module_.posterior.view._unflatten(sample)
                for i, sample in pl_module_.sample_container.items()
            }
        else:
            res = pl_module_.sample_container.samples

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

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:

        assert isinstance(pl_module, MCMCInference)
        assert isinstance(pl_module.sampler, SGHMCWithVarianceEstimator)

        with torch.random.fork_rng():
            torch.manual_seed(123)
            self.log_idx, _ = torch.sort(
                torch.randperm(pl_module.posterior.shape[0])[: self.n_gradients]
            )
            self.estimates: Dict[int, Dict[str, Tensor]] = {}

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: BATCH_IN,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:

        pl_module = cast(MCMCInference, pl_module)

        if trainer.global_step % self.steps_per_log == 0:
            self.log_gradient_estimate(trainer, pl_module, batch_idx)

    def log_gradient_estimate(
        self, trainer: Trainer, pl_module: MCMCInference, batch_idx: int
    ) -> None:

        sampler = cast(SGHMCWithVarianceEstimator, pl_module.sampler)

        with torch.random.fork_rng():

            wf_estimator = WelfordEstimator(pl_module.posterior.shape)
            for _ in range(self.estimation_steps):
                x, y = next(iter(trainer.train_dataloader))
                sampling_fraction = len(x) / len(
                    cast(Sized, trainer.train_dataloader.dataset)
                )
                with pl_module.posterior.observe(x, y, sampling_fraction):
                    wf_estimator.update(pl_module.posterior.grad_prop_log_p())

        observed_variance = wf_estimator.estimate()
        estimated_variance = sampler.variance_estimator.estimate()

        observed_variance, estimated_variance = torch.broadcast_tensors(
            observed_variance, estimated_variance
        )

        self.estimates[trainer.global_step] = {
            "observed_variance": observed_variance,
            "estimated_variance": estimated_variance,
        }

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        torch.save({"log_idx": self.log_idx, "estimates": self.estimates}, self.path)


class SGHMCLogTemperature(Callback):
    def __init__(
        self, steps_per_log: int = 50, path: Union[str, Path] = "temperature_samples.pt"
    ) -> None:

        self.path = path
        self.temperature_samples: Dict[Tuple[int, str], Dict[str, Number]] = {}
        self.steps_per_log = steps_per_log

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        assert isinstance(pl_module, MCMCInference)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: BATCH_IN,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:

        if trainer.global_step % self.steps_per_log != 0:
            return

        pl_module = cast(MCMCInference, pl_module)

        if isinstance(pl_module.sampler, SGHMCWithVarianceEstimator):
            M_diag = pl_module.sampler.mass_factor
            step_size = torch.sqrt(pl_module.sampler.lr_0)
            nu = pl_module.sampler.nu
            r = nu / step_size * M_diag
            norm_squares = (1 / M_diag) * r * r
        elif isinstance(pl_module.sampler, SGHMC):
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

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        torch.save(self.temperature_samples, self.path)


class GetSampleFilterCurve(Callback):
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        assert isinstance(pl_module, MCMCInference)

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:

        pl_module = cast(MCMCInference, pl_module)

        sample_idx_order = torch.randperm(
            len(pl_module.sample_container), generator=torch.Generator().manual_seed(24)
        ).tolist()
        sample_idx = list(pl_module.sample_container.keys())
        self.samples_order = [sample_idx[i] for i in sample_idx_order]
        self.results: list[Dict[str, Number]] = []

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: BATCH_IN,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:

        _, y = batch
        pred = torch.tensor(0)

        if outputs is None or not isinstance(outputs, dict):
            return

        per_sample_predictions: Dict[int, Tensor] = outputs["per_sample_predictions"]
        for i, sample_idx in enumerate(self.samples_order):
            pred = pred + per_sample_predictions[sample_idx]
            error_rate = 1 - FM.accuracy(pred, y)
            self.results.append(
                {
                    "batch_idx": batch_idx,
                    "with_n_samples": i + 1,
                    "error_rate": error_rate.item(),
                }
            )

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        stats: pd.DataFrame = (
            pd.DataFrame(self.results)
            .set_index("batch_idx")
            .groupby("with_n_samples")
            .mean("error_rate")
        )
        stats.to_json("sample_resampling_curve.json")
