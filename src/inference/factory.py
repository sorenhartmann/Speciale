from dataclasses import dataclass, field
from typing import Dict, Union
from src.inference.mcmc.mcmc import MCMCInference, StochasticGradientHamiltonian
from src.inference.mcmc.sample_containers import FIFOSampleContainer
from src.inference.sgd import SGDInference
from src.inference.vi import VariationalInference

from src.models.mlp import MLPClassifier

from src.utils import RegisteredComponents


@dataclass
class ComponentConfig:

    name: str
    config: Dict[str, Union["ComponentConfig", float, int]] = field(
        default_factory=dict
    )

def instantiate_from_config(config: ComponentConfig):

    kwargs = {}
    for k, v in config.config.items():
        if type(v) is ComponentConfig:
            kwargs[k] = instantiate_from_config(v)
        else:
            kwargs[k] = v

    return RegisteredComponents.components[config.name](**kwargs)

def from_flat_config(root_name, flat_config):
        
    split_iter = ((k.split("."), v) for  k, v in flat_config.items())
    adj_dict = {}
    for (x, y), z in split_iter:
        if x not in adj_dict:
            adj_dict[x] = {}
        adj_dict[x][y] = z

    def _sub(name):
        config = {}
        for y, z in adj_dict[name].items():
            if z in adj_dict:
                config[y] = _sub(z)
            else:
                config[y] = z
        return ComponentConfig(name, config)

    return _sub(root_name)


specs = [
    {
    "model": "mlp_classifier",
    "inference": "sgd",
    "sgd.lr": 0.005,
    },
    {
    "model": "mlp_classifier",
    "inference": "vi",
    "vi.lr": 1e-3,
    "vi.n_samples": 10,
    "vi.prior": None,  # Not implemented
    },
    {
    "model": "mlp_classifier",
    "inference": "mcmc",
    "mcmc.sampler": "sghmc",
    "mcmc.sample_container": "fifo",
    "mcmc.burn_in": 50,
    "sghmc.lr": 0.2e-5,
    "fifo.max_items": 10,
    "fifo.keep_every": 1,
    }
]

for spec in specs:
    flat_config = spec.copy()
    model_name = flat_config.pop("model")
    inference_name = flat_config.pop("inference")
    inference_config = from_flat_config(inference_name, flat_config)
    inference_config.config.update({"model" : ComponentConfig(name=model_name)})
    inf = instantiate_from_config(inference_config)
    print(inf)


# ComponentConfig(
#             name="mcmc",
#             config={
#                 "sampler": ComponentConfig(name="sghmc", config={"lr": 50}),
#                 "sample_container": ComponentConfig(
#                     name="fifo", config={"max_items": 10, "keep_every": 1}
#                 ),
#                 "burn_in": 50,
# #               "model": ComponentConfig(name="mlp_classifer"),
#             },
#         )



# def _unflatten_config_sub(name, sub_flat_config):
#     pass

# def unflatten_config(name, flat_config):

#     components = {k.split(".")[0] for k in flat_config}
#     split_config = ((k.split("."), v) for k, v in flat_config.items() )
#     flat_config_dict = {x : (y, z) for (x, y), z in split_config }
    


# def from_flat_config(name, flat_config):
    
#     flat_config.

#     flat_spec_dict = {}



    
#     pass



# def _from_flat_config(name, flat_config):

#     config = {}
#     keys = {x.split(".")[0] for x in flat_config.keys()}
#     for key in keys:
#         sub_config = {k: v for k, v in flat_config.items() if k.startswith(key)}
#         if len(sub_config) == 1:
#             config[key] = sub_config[key]
#         else:
#             sub_config_name = sub_config.pop(key)
#             sub_config = {k[len(key) + 1 :]: v for k, v in sub_config.items()}
#             sub_config = _from_flat_config(name=sub_config_name, flat_config=sub_config)

#             config[key] = sub_config

#     return ComponentConfig(name=name, config=config)
