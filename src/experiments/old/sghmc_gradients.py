from functools import cache
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from pytorch_lightning import Callback

from src.experiments.common import plot, result
from src.inference.mcmc.variance_estimators import (VarianceEstimator,
                                               WelfordEstimator)


class DummyVarianceEstimator(VarianceEstimator):
    def __init__(self, variance_estimator: VarianceEstimator, use_estimate, constant=0.0):

        super().__init__()
        self.use_estimate = use_estimate
        self.constant = torch.tensor(constant)
        self.wrapped = variance_estimator

    def setup(self, sampler):
        self.wrapped.setup(sampler)

    def estimate(self) -> torch.Tensor:
        if self.use_estimate:
            return self.wrapped.estimate()
        else:
            return self.constant

    def on_train_epoch_start(self, inference_module):
        self.wrapped.on_train_epoch_start(inference_module)

    def on_after_grad(self, grad: torch.Tensor):
        self.wrapped.on_after_grad(grad)

    def on_before_next_sample(self, sampler):
        self.wrapped.on_before_next_sample(sampler)


class LogVarianceEstimates(Callback):

    interbatch_variance_folder = Path("variance_interbatch")
    variance_estimated_folder = Path("variance_estimated")

    def __init__(self, n_gradients=1000, logs_per_epoch=10):

        self.n_gradients = n_gradients
        self.logs_per_epoch = logs_per_epoch

    def on_init_start(self, trainer) -> None:

        self.interbatch_variance_folder.mkdir()
        self.variance_estimated_folder.mkdir()

    def _get_estimate(self, estimator):
        if type(estimator) is VarianceEstimatorWrapper:
            return estimator.wrapped.estimate()
        else:
            return estimator.estimate()

    def on_fit_start(self, trainer, pl_module) -> None:

        with torch.random.fork_rng():
            torch.manual_seed(123)
            self.log_idx, _ = torch.sort(
                torch.randperm(pl_module.posterior.shape[0])[: self.n_gradients]
            )
        torch.save(self.log_idx, "log_idx.pt")

    @cache
    def sample_steps(self, trainer):
        num_batches = len(trainer.train_dataloader)
        steps_between_samples = num_batches / self.logs_per_epoch
        return [
            int(i * steps_between_samples) - 1
            for i in range(1, self.logs_per_epoch + 1)
        ]

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, unused
    ) -> None:

        if batch_idx in self.sample_steps(trainer):
            self.log_gradient_estimate(trainer, pl_module, batch_idx)

    def log_gradient_estimate(self, trainer, pl_module, batch_idx):

        with torch.random.fork_rng():

            wf_estimator = WelfordEstimator()
            for batch in trainer.train_dataloader:
                x, y = batch
                sampling_fraction = len(x) / len(trainer.train_dataloader.dataset)
                with pl_module.posterior.observe(x, y, sampling_fraction):
                    wf_estimator.update(pl_module.posterior.grad_prop_log_p())

        variance = wf_estimator.estimate()
        estimate = self._get_estimate(pl_module.sampler.variance_estimator)

        variance, estimate = torch.broadcast_tensors(variance, estimate)

        torch.save(
            variance[self.log_idx],
            self.interbatch_variance_folder / f"{trainer.global_step:06}.pt",
        )
        torch.save(
            estimate[self.log_idx],
            self.variance_estimated_folder / f"{trainer.global_step:06}.pt",
        )


@result
def variance_estimates():

    variance_interbatch = []
    for file in sorted(Path("variance_interbatch").iterdir()):
        variance_interbatch.append(torch.load(file))

    variance_estimated = []
    for file in sorted(Path("variance_estimated").iterdir()):
        variance_estimated.append(torch.load(file))

    return {
        "variance_interbatch": torch.stack(variance_interbatch),
        "variance_estimated": torch.stack(variance_estimated),
    }


@result
def global_steps():
    return [int(file.stem) for file in sorted(Path("variance_interbatch").iterdir())]


@plot(multirun=True)
def final_estimates(variance_estimates, _run_):

    plot_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "Using estimate": run.get_override_value(
                        "inference.sampler.variance_estimator.use_estimate"
                    ),
                    "Estimator": run.get_override_value(
                        "experiment/sghmc_gradients/estimator@estimator"
                    ),
                    "Interbatch variance": x["variance_interbatch"][-1],
                    "Estimated variance": x["variance_estimated"][-1],
                }
            )
            for run, x in zip(_run_.values(), variance_estimates.values())
        ]
    )

    fg = sns.relplot(
        data=plot_data,
        x="Inter batch variance",
        y="Estimated variance",
        col="Estimator",
        row="Using estimate",
    )
    fg.set(xscale="log")
    fg.set(yscale="log")
    for ax in fg.axes.flatten():
        ax.axline((0, 0), (1, 1), color="C1")

    plt.savefig("final_estimates.pdf")


@plot(multirun=False)
def all_estimates_sampled_variables(variance_estimates, global_steps, _run_):

    is_zero = variance_estimates["variance_estimated"].isclose(torch.tensor(0.0)).all(0)
    is_zero |= (
        variance_estimates["variance_interbatch"].isclose(torch.tensor(0.0)).all(0)
    )

    non_zero_index = (~is_zero).nonzero().flatten()

    with torch.random.fork_rng():
        torch.manual_seed(123)
        plot_idx = non_zero_index[torch.randperm(len(non_zero_index))[:9]]

    def get_data(idx):

        return pd.DataFrame(
            {
                "Inter batch variance": variance_estimates["variance_interbatch"][
                    :, idx
                ],
                "Estimated variance": variance_estimates["variance_estimated"][:, idx],
            },
            index=pd.MultiIndex.from_product(
                [[idx.item()], global_steps], names=("index", "step")
            ),
        )

    fg = (
        pd.concat(map(get_data, plot_idx))
        .assign(step_mod_100=lambda x: x.index.get_level_values("step") % 100)
        .reset_index()
        .sample(frac=1.0)
        .pipe(
            (sns.relplot, "data"),
            x="Inter batch variance",
            y="Estimated variance",
            col="index",
            col_wrap=3,
            hue="step_mod_100",
            facet_kws={"sharey": False, "sharex": False},
        )
    )

    fg.set(xscale="log")
    fg.set(yscale="log")
    for ax in fg.axes.flatten():
        ax.axline((0, 0), (1, 1), color="red")

    plt.savefig("all_estimates.pdf")



# class LogGradientVariance(Callback):

#     def __init__(self):
#         self.gradient_vars = []

#     def on_train_epoch_end(self, trainer: Trainer, pl_module) -> None:

#         gradients = []
#         with torch.random.fork_rng():
#             for batch in trainer.train_dataloader:
#                 x, y = batch
#                 sampling_fraction = len(x) / len(trainer.train_dataloader.dataset)
#                 with pl_module.posterior.observe(x, y, sampling_fraction):
#                     gradients.append(pl_module.posterior.grad_prop_log_p())

#             variance = torch.var(torch.stack(gradients), 0, unbiased=True)
#             pl_module.log("avg_grad_variance", variance.mean().item())

#         self.gradient_vars.append(variance)

#         if trainer.current_epoch % 50 == 0:

#             variance_images = pl_module.posterior.unflatten(variance)
#             for name, img in variance_images.items():
#                 shape = img.shape
#                 shape = (1,) * (3 - len(shape)) + shape
#                 trainer.logger.experiment.add_image(
#                     f"gradient_variance/{name}",
#                     img.view(*shape),
#                     global_step=trainer.current_epoch,
#                 )

#             trainer.logger.experiment.add_histogram(
#                 "gradient_variance", variance, global_step=trainer.current_epoch
#             )

#     def on_fit_end(self, trainer, pl_module) -> None:
#         torch.save(self.gradient_vars, "./gradient_vars.pkl")

# class LogGradients(Callback):

#     def __init__(self, logged_fraction=1, log_runs=1):

#         self.logged_fraction = logged_fraction
#         self.log_runs = log_runs
#         self.save_dir = Path("./logged_gradients")
#         self.save_dir.mkdir()

#     def on_train_start(self, trainer, pl_module) -> None:

#         interval_length = trainer.max_epochs * self.logged_fraction / self.log_runs

#         gap = (trainer.max_epochs - trainer.max_epochs * self.logged_fraction) / (
#             max(self.log_runs - 1, 1)
#         )

#         start = 0
#         end = interval_length

#         self.intervals = []
#         for _ in range(self.log_runs):
#             self.intervals.append((int(start), int(end)))
#             start = end + gap
#             end = start + interval_length

#     def on_train_epoch_start(self, trainer, pl_module) -> None:

#         e = trainer.current_epoch

#         if any( a <= e <= b for a, b in self.intervals):
#             self.gradients = []
#         else:
#             self.gradients = None

#     def on_batch_end(self, trainer, pl_module) -> None:

#         if self.gradients is None:
#             return

#         self.gradients.append(pl_module.posterior.state_grad.clone())

#     def on_train_epoch_end(self, trainer, pl_module) -> None:

#         if self.gradients is None:
#             return

#         torch.save(
#             torch.stack(self.gradients),
#             self.save_dir / f"gradients_epoch_{trainer.current_epoch:04}.pt",
#         )

#         del self.gradients
