from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from hydra.utils import instantiate
from pytorch_lightning import Callback, Trainer

from src.experiments.common import Experiment, Run
from src.inference.mcmc.var_estimators import (VarianceEstimator,
                                               WelfordEstimator)


def plot(func):
    func.__isplot = True
    return func

def result(func):
    func.__isresult = True
    return func

class VarianceEstimatorWrapper(VarianceEstimator):

    def __init__(self, variance_estimator, use_estimate, constant=0.):
        
        super().__init__()
        self.use_estimate = use_estimate
        self.constant = torch.tensor(constant)
        self.variance_estimator = variance_estimator


    def setup(self, sampler):
        self.variance_estimator.setup(sampler)

    def update(self, grad: torch.Tensor):
        self.variance_estimator.update(grad)

    def estimate(self) -> torch.Tensor:
        if self.use_estimate:
            return self.variance_estimator.estimate()
        else:
            return self.constant

class LogVarianceEstimates(Callback):

    inter_batch_variance_folder = Path("variance_inter_batch")
    variance_estimated_folder = Path("variance_estimated")

    def on_init_start(self, trainer) -> None:

        self.inter_batch_variance_folder.mkdir()
        self.variance_estimated_folder.mkdir()

    def _get_estimate(self, estimator):
        if type(estimator) is VarianceEstimatorWrapper:
            return estimator.variance_estimator.estimate()
        else:
            return estimator.estimate()

    # def on_batch_end(self, trainer, pl_module) -> None:

    #     estimator = pl_module.sampler.variance_estimator

    #     var_est = self._get_estimate(estimator)

    #     pl_module.log("grad_var_est/max", var_est.max())
    #     pl_module.log("grad_var_est/min", var_est.min())

    def on_train_epoch_end(self, trainer: Trainer, pl_module) -> None:

        with torch.random.fork_rng():
            wf_estimator = WelfordEstimator()
            for batch in trainer.train_dataloader:
                x, y = batch
                sampling_fraction = len(x) / len(trainer.train_dataloader.dataset)
                with pl_module.posterior.observe(x, y, sampling_fraction):
                    wf_estimator.update(pl_module.posterior.grad_prop_log_p())

            variance = wf_estimator.estimate()
            
            pl_module.log("grad_var/max", variance.max())
            pl_module.log("grad_var/min", variance.min())

            estimate = self._get_estimate(pl_module.sampler.variance_estimator)

            torch.save(
                variance,
                self.inter_batch_variance_folder / f"{trainer.current_epoch:04}.pt",
            )
            torch.save(
                estimate,
                self.variance_estimated_folder / f"{trainer.current_epoch:04}.pt",
            )


# Kunne give run/run_collection til functionen her
@result
def variance_estimates_sample(cfg):

    experiment = Experiment(cfg.experiment)

    if cfg.run is None:
        exp_df = experiment.as_dataframe()
        latest_run_dir = exp_df.loc[lambda x: ~x.path.isna()].iloc[-1].path
        run = Run(latest_run_dir)

    sample_idx = None

    variance_estimated = []
    for file in (run.path / "variance_estimated").iterdir():
        variance_estimated_ = torch.load(file)
        if sample_idx is None:
            sample_idx = torch.randint(len(variance_estimated_), (100,))
        variance_estimated.append(variance_estimated_[sample_idx])
    variance_estimated = torch.stack(variance_estimated)

    variance_inter_batch = []
    for file in (run.path / "variance_inter_batch").iterdir():
        variance_inter_batch_ = torch.load(file)
        variance_inter_batch.append(variance_inter_batch_[sample_idx])
    variance_inter_batch = torch.stack(variance_inter_batch)

    return {
        "variance_inter_batch" : variance_inter_batch,
        "variance_estimated" : variance_estimated,
    }



# plot kun hvor var > 1e-5 eller noget..
@plot
def plot_estimates(variance_estimates_sample):

    variance_inter_batch = variance_estimates_sample["variance_inter_batch"]
    variance_estimated = variance_estimates_sample["variance_estimated"]
    
    plt.scatter(variance_inter_batch[0], variance_estimated[0])
    plt.axline((0, 0), slope=1.0, color="C1")
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig("estimates.pdf")


@plot
def plot_estimates_over_time(variance_estimates_sample):

    variance_inter_batch = variance_estimates_sample["variance_inter_batch"]
    variance_estimated = variance_estimates_sample["variance_estimated"]

    torch.manual_seed(123)
    sample_idx = torch.randint(variance_inter_batch.shape[1], (10,))

    plt.plot(
        variance_inter_batch[50:70,sample_idx],
        variance_estimated[50:70,sample_idx],
    )
    
    plt.axline((0, 0), slope=1.0, color="C1")

    plt.xscale("log")
    plt.yscale("log")
    plt.savefig("estimates_over_time.pdf") #TODO: move to outer call



@hydra.main("../../conf", "experiment/sghmc_gradients/config")
def experiment(cfg):

    torch.manual_seed(123)

    dm = instantiate(cfg.data)

    inference = instantiate(cfg.inference)
    trainer = instantiate(cfg.trainer)

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
