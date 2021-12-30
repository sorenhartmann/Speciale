from pathlib import Path
from typing import Any, Dict, cast

import hydra
import optuna
import torch
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf
from optuna import Trial
from pytorch_lightning import Trainer

from src.data.data_module import DataModule
from src.inference.base import InferenceModule
from src.utils import set_directory


def get_search_grid(search_space: DictConfig) -> Dict[str, Any]:
    search_space_ = cast(Dict[str, Any], search_space)  # Hydra shenanigans
    return {k: v["_grid_options_"] for k, v in search_space_.items()}


def get_sqlite_storage_string(study_name: str) -> str:
    return f"sqlite:///{get_original_cwd()}/optuna_storages/{study_name}.db"


OmegaConf.register_new_resolver("get_search_grid", get_search_grid)
OmegaConf.register_new_resolver("get_sqlite_storage_string", get_sqlite_storage_string)


def get_suggestions(trial: Trial, search_space_cfg: DictConfig) -> Dict[str, Any]:

    suggestions = {}
    search_space_cfg_ = cast(
        Dict[str, DictConfig], search_space_cfg
    )  # Hydra shenanigans
    for name, config in search_space_cfg_.items():
        config_: Dict[str, Any] = dict(config)
        suggest_method_name = f"suggest_{config_.pop('type')}"
        suggest_method = getattr(trial, suggest_method_name)
        if "_grid_options_" in config_:
            del config_["_grid_options_"]
        suggestions[name] = suggest_method(name=name, **config_)

    return suggestions


class TrainingError(Exception):
    ...


@hydra.main("../conf", "sweep_config")
def main(cfg: DictConfig) -> None:
    def objective(trial: Trial) -> float:

        work_dir = (Path(".") / f"{trial.number:03}").resolve()
        work_dir.mkdir()

        with set_directory(work_dir):

            trial_cfg = cfg.copy()
            suggestions = get_suggestions(trial, cfg.sweep.search_space)
            for key, suggestion in suggestions.items():
                OmegaConf.update(trial_cfg, key, suggestion)

            config_path = Path(".hydra/config.yaml")
            config_path.parent.mkdir()
            OmegaConf.save(trial_cfg, config_path)

            torch.manual_seed(trial_cfg.seed)

            dm: DataModule = instantiate(trial_cfg.data)
            inference: InferenceModule = instantiate(trial_cfg.inference)

            # add extra callbacks
            _callbacks = list(cfg.trainer.get("callbacks", []))
            _callbacks += list(cfg.get("extra_callbacks", []))
            callbacks = [instantiate(x) for x in _callbacks]
            callbacks += [instantiate(cfg.optuna_callback, trial=trial)]

            trainer: Trainer = instantiate(trial_cfg.trainer, callbacks=callbacks)

            try:
                trainer.fit(inference, dm)
            except ValueError:
                raise optuna.TrialPruned
            except:
                raise

        if trainer.interrupted:
            raise KeyboardInterrupt

        return cast(float, trainer.logged_metrics.get(cfg.sweep.monitor))

    sampler: optuna.samplers.BaseSampler = instantiate(cfg.sweep.sampler)
    study: optuna.Study = instantiate(cfg.sweep.study, sampler=sampler)

    study.optimize(objective, **cfg.sweep.optimize_args)


if __name__ == "__main__":
    main()
