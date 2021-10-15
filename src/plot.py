

import hydra
from inspect import getmembers, isfunction, signature
from importlib import import_module
from src.experiments.common import Experiment

@hydra.main("../conf/plot", "main")
def main(cfg):

    experiment_module = import_module(f"src.experiments.{cfg.experiment}")
    members = getmembers(experiment_module)

    plot_funcs = {name: x for name, x in members if getattr(x, "__isplot", False)}
    result_funcs =  {name: x for name, x in members if getattr(x, "__isresult", False)}

    results = {}
    for func in plot_funcs.values():
        args = {}
        for arg_name in signature(func).parameters:
            if arg_name not in results:
                result = result_funcs[arg_name](cfg)
                results[arg_name]  = result
            args[arg_name] = results[arg_name] 
        func(**args)



if __name__ == "__main__":



    main()