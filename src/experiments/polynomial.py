
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from src.data.polynomial import PolynomialDataModule
from src.experiments.common import (ExperimentHandler, FlatCSVLogger, Run,
                                    get_args)
from src.inference import BayesianRegressor
from src.models.polynomial import PolynomialModel


class LogPosteriorSamples(Callback):
    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):

        if pl_module.global_step < pl_module.burn_in:
            # Burn in sample
            return

        if (pl_module.burn_in + pl_module.global_step) % pl_module.use_every != 0:
            # Thin sample
            return

        for k, param in pl_module.model.named_parameters():
            for i, value in enumerate(param.flatten()):
                pl_module.log(f"weights/{k}.{i}", value)

def experiment():

    args = get_args(PolynomialModel, BayesianRegressor)

    seed = 123
    sampler = args.sampler.from_argparse_args(args)
    dm = PolynomialDataModule(args.batch_size, train_obs=1000, val_obs=100)
    model = PolynomialModel(torch.randn(4))
    inference = BayesianRegressor.from_argparse_args(args, model=model, sampler=sampler)

    logger = FlatCSVLogger(".")
    ckpt_callback = ModelCheckpoint("./checkpoints")
    
    trainer = Trainer.from_argparse_args(
        args,
        callbacks=[LogPosteriorSamples(), ckpt_callback],
        logger=logger,
    )

    conf = dict(inference.hparams)
    conf["seed"] = seed

    torch.manual_seed(seed)
    trainer.fit(inference, dm)

def plot_posterior_samples(run=None, ax=None):

    if run is None:
        run = Run(".")

    if ax is None:
        ax = plt.gca()

    fig = plt.gcf()

    metrics = run.metrics.droplevel("epoch")

    weight_cols = metrics.columns.to_series().loc[lambda x: x.str.match("weights/")]
    weights_samples = metrics[weight_cols]

    plot_data = weights_samples.reset_index().melt(id_vars="step")
    grid_spec = fig.add_gridspec(2, 1, height_ratios=(2, 7))

    ax_line = fig.add_subplot(grid_spec[1, 0])
    ax_marg = fig.add_subplot(grid_spec[0, 0], sharex=ax_line)

    sns.lineplot(
        x="value",
        y="step",
        hue="variable",
        ax=ax_line,
        data=plot_data,
        sort=False,
        legend=False,
    )
    sns.kdeplot(x="value", hue="variable", data=plot_data, legend=False)

    plt.show()



def main():

    handler = ExperimentHandler(experiment)
    handler.run()

if __name__ == "__main__":

    main()
