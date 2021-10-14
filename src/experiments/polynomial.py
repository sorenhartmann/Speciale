import hydra
import matplotlib.pyplot as plt
import seaborn as sns
import torch


@hydra.main("../../conf", "experiment/polynomial/config")
def experiment(cfg):

    seed = 123

    model = hydra.utils.instantiate(cfg.model)
    datamodule = hydra.utils.instantiate(cfg.data, model=model)
    for coeff in model.parameters():
        with torch.no_grad():
            coeff.normal_()
    inference = hydra.utils.instantiate(cfg.inference, model=model)
    trainer = hydra.utils.instantiate(cfg.trainer)

    trainer.fit(inference, datamodule)
    print("fdgfd")




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


if __name__ == "__main__":

    experiment()
