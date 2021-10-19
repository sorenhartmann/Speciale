import hydra
from hydra.utils import instantiate

from src.experiments.common import EXPERIMENT_PATH, Experiment
from omegaconf import OmegaConf


def latest_run_dir(name):
    run_dirs = Experiment(name).run_dirs()
    latest_dir = run_dirs[-1]
    return str(latest_dir.relative_to(EXPERIMENT_PATH))


OmegaConf.register_new_resolver("latest_run_dir", latest_run_dir)

import logging

log = logging.getLogger(__name__)


@hydra.main("../conf/plot", "main")
def main(cfg):

    run = instantiate(cfg.run)
    log.info(
        f"Generating plots for \"{cfg.experiment_name}\", run at {run.time}"
    )

    plot_funcs = run.experiment.get_plot_funcs()
    for plot_func in plot_funcs.values():
        # for arg_name in signature(plot_func).parameters:
        run.call_plot_func(plot_func)


if __name__ == "__main__":

    main()
