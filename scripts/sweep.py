import os
import hydra
import torch
from hydra.utils import instantiate, to_absolute_path, get_original_cwd
from contextlib import contextmanager
from pathlib import Path
import optuna
from omegaconf import OmegaConf
from pytorch_lightning import Callback

@contextmanager
def set_directory(path: Path):

    origin = Path().absolute()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(origin)

def get_search_grid(search_space):
    return {k : v["_grid_options_"] for k,v in search_space.items()}

OmegaConf.register_new_resolver("get_search_grid", get_search_grid)

def get_suggestions(trial, search_space_cfg):

    suggestions = {}
    for name, config in search_space_cfg.items():
        config = dict(config)
        suggest_method_name = f"suggest_{config.pop('type')}"
        suggest_method = getattr(trial, suggest_method_name)
        if "_grid_options_" in config:
            del config["_grid_options_"]
        suggestions[name] = suggest_method(name=name, **config)
    return suggestions

class TrainingError(Exception):
    ...


@hydra.main("../conf", "sweep_config")
def main(cfg):

    def objective(trial: optuna.Trial):

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

            dm = instantiate(trial_cfg.data)
            inference = instantiate(trial_cfg.inference)
            callback_cfg =  trial_cfg.trainer.get("callbacks")
            if callback_cfg is not None:
                callbacks = [instantiate(x) for x in trial_cfg.trainer.callbacks]
            else:
                callbacks = []
            callbacks += [instantiate(cfg.optuna_callback, trial=trial)]

            trainer = instantiate(trial_cfg.trainer, callbacks=callbacks)
            
            try:
                trainer.fit(inference, dm)
            except ValueError:
                raise TrainingError
            except:
                raise

        if trainer.interrupted:
            raise KeyboardInterrupt

        return trainer.logged_metrics.get(cfg.sweep.monitor)

    storage = f"sqlite:///{get_original_cwd()}/{cfg.sweep.study_storage_file_name}"
    sampler = instantiate(cfg.sweep.sampler)
    study = instantiate(cfg.sweep.study, storage=storage, sampler=sampler)
    
    study.optimize(objective, catch=(TrainingError,), **cfg.sweep.optimize_args)


if __name__ == "__main__":
    main()
