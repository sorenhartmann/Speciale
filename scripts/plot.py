import hydra
from hydra.utils import instantiate, call

from src.experiments.common import EXPERIMENT_PATH, Experiment
from omegaconf import OmegaConf

import logging

log = logging.getLogger(__name__)


def latest_run_dir(name):
    run_dirs = Experiment(name).run_dirs()
    latest_dir = run_dirs[-1]
    return str(latest_dir.relative_to(EXPERIMENT_PATH))

OmegaConf.register_new_resolver("latest_run_dir", latest_run_dir)

@hydra.main("../conf", "plotting_config")
def main(cfg):
    run = instantiate(cfg.run)
    for name, plot_cfg in cfg.plot.items():
        msg = f'Generating "{name}" plot for "{cfg.experiment_name}" run at {run.time}'
        log.info(msg)
        call(plot_cfg, run)

if __name__ == "__main__":
    main()
