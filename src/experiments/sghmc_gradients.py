import hydra
import torch
from hydra.utils import instantiate
from pytorch_lightning import Callback, Trainer
from src.inference.mcmc.samplers import VarianceEstimator
import seaborn as sns
import matplotlib.pyplot as plt


class ZeroVarianceEstimator(VarianceEstimator):

    def estimate(self):
        return torch.tensor(0.0)

    def estimate_(self):
        return super().estimate()

class LogVarianceEstimates(Callback):

    def _get_estimate(self, estimator):
        if type(estimator) is ZeroVarianceEstimator:
            return estimator.estimate_()
        else:
            return estimator.estimate()
    

    def on_batch_end(self, trainer, pl_module) -> None:
        estimator = pl_module.sampler.var_estimator

        var_est = self._get_estimate(estimator)

        pl_module.log("grad_var_est/max", var_est.max())
        pl_module.log("grad_var_est/min", var_est.min())

    def on_train_epoch_end(self, trainer: Trainer, pl_module) -> None:

        gradients = []
        with torch.random.fork_rng():
            for batch in trainer.train_dataloader:
                x, y = batch
                sampling_fraction = len(x) / len(trainer.train_dataloader.dataset)
                with pl_module.posterior.observe(x, y, sampling_fraction):
                    gradients.append(pl_module.posterior.grad_prop_log_p())

            variance = torch.var(torch.stack(gradients), 0, unbiased=True)
            pl_module.log("grad_var/max", variance.max())
            pl_module.log("grad_var/min", variance.min())

            if trainer.current_epoch % 5 == 0:
                estimate = self._get_estimate(pl_module.sampler.var_estimator)
                non_zero = variance > 0

                clamp = pl_module.sampler.var_estimator.clamp
                adj_with_mean = pl_module.sampler.var_estimator.adj_with_mean

                sns.relplot(x="Variance", y="Estimate", data={
                    "Estimate": estimate[non_zero].numpy(),
                    "Variance": variance[non_zero].numpy()
                })
                plt.title(f"{clamp=},{adj_with_mean=}")
                plt.axline((0, 0), slope=1.0, color="red")
                plt.xscale("log")
                plt.yscale("log")

                pl_module.logger.experiment.add_figure(
                    "grad_var_est_comp", plt.gcf(), trainer.current_epoch
                )
                plt.close()


import warnings

@hydra.main("../../conf", "experiment/sghmc_gradients/config")
def experiment(cfg):

    torch.manual_seed(123)

    dm = instantiate(cfg.data)

    inference = instantiate(cfg.inference)
    trainer = instantiate(cfg.trainer)

    with warnings.catch_warnings():

        trainer.fit(inference, dm)


if __name__ == "__main__":

    experiment()


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
