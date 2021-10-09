import pytorch_lightning as pl


class InferenceModule(pl.LightningModule):
    ...
    
#     # Burde nok gøres med tb
#     def save_hyperparameters(self, path=None):
#         hparams = get_hyperparameters(self)
#         hparams = {
#             k: x.item() if isinstance(x, torch.Tensor) else x
#             for k, x in hparams.items()
#         }

#         if path is None:
#             path = Path("./hparams.yaml")

#         with open(path, "w") as f:
#             yaml.dump(hparams, f)

# def get_component_name(component):

#     component_names = {y: x for x, y in RegisteredComponents.components.items()}

#     try:
#         return component_names[component.__class__]
#     except KeyError:
#         pass

#     return component_names[component.__class__.__base__]


# def _iter_component_hparams(component, prefix=""):

#     for name, type_ in type(component).__annotations__.items():

#         if not getattr(type_, "__origin__", None) is HPARAM:
#             continue

#         param_type = type_.__args__[0]

#         if param_type is Component:

#             sub_component = getattr(component, name)
#             sub_component_name = get_component_name(sub_component)

#             if isinstance(sub_component, Model):
#                 continue

#             yield f"{prefix}{name}", sub_component_name
#             yield from _iter_component_hparams(
#                 sub_component, prefix=f"{sub_component_name}."
#             )
#             # yield from ("{name}.{x}" for x in iter_hyperparameters(getattr(component, name)))

#         elif getattr(param_type, "__origin__", None) is list:

#             param_list = getattr(component, name)
#             for i, x in enumerate(param_list):

#                 if type(x) is Component:
#                     raise NotImplementedError
#                 else:
#                     yield f"{name}.{i}", x

#         else:

#             value = getattr(component, name)
#             yield f"{prefix}{name}", value


# def get_hyperparameters(inference):

#     inference_name = get_component_name(inference)
#     hparams = {
#         "model": get_component_name(inference.model),
#         "inference": inference_name,
#     }
#     hparams.update(
#         dict(_iter_component_hparams(inference, prefix=f"{inference_name}."))
#     )
#     return hparams
