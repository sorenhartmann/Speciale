import hydra
import torch
from hydra.utils import instantiate
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import pandas as pd
from src.experiments.common import result, plot


@hydra.main("../../conf", "experiment/mnist/config")
def experiment(cfg):

    torch.manual_seed(123)
    dm = instantiate(cfg.data)

    inference = instantiate(cfg.inference)
    trainer = instantiate(cfg.trainer)

    trainer.fit(inference, dm)

    return trainer.callback_metrics["err/val"].item()


@result
def model_performance():

    accumulator = EventAccumulator("metrics")
    accumulator.Reload()

    def get_series(tag):
        index = [x.step for x in accumulator.Scalars(tag)]
        value = [x.value for x in accumulator.Scalars(tag)]
        return pd.Series(value, index=index)

    return pd.DataFrame(
        {
            "Epoch": get_series("epoch"),
            "Validation error": get_series("err/val"),
        }
    )


import seaborn as sns
import matplotlib.pyplot as plt


@plot(multirun=True)
def validation_curves(model_performance, _run_):

    (
        pd.concat(
            x.assign(i=str(i))
            for i, x in model_performance.items()
        )
        .reset_index()
        .pipe(
            (sns.relplot, "data"),
            x="Epoch",
            y="Validation error",
            hue="i",
            kind="line"
        )
    )

    plt.savefig("validation_curves.pdf")


if __name__ == "__main__":

    experiment()
