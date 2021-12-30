from torch.nn import BatchNorm2d, Conv2d, Linear

from src.bayesian.core import BayesianConversionConfig, BayesianModuleConfig
from src.bayesian.modules import BayesianConv2d, BayesianLinear, BayesianNop
from src.bayesian.priors import NormalPrior, ScaleMixturePrior


def get_normal(
    mean: float = 0.0,
    precision: float = 1.0,
) -> BayesianConversionConfig:
    return BayesianConversionConfig(
        modules_to_replace={
            Linear: BayesianModuleConfig(
                module=BayesianLinear,
                priors={
                    "weight": NormalPrior(precision=precision, mean=mean),
                    "bias": NormalPrior(precision=precision, mean=mean),
                },
            ),
            Conv2d: BayesianModuleConfig(
                module=BayesianConv2d,
                priors={
                    "weight": NormalPrior(precision=precision, mean=mean),
                    "bias": NormalPrior(precision=precision, mean=mean),
                },
            ),
            BatchNorm2d: BayesianModuleConfig(
                module=BayesianNop,
                priors={},
            ),
        }
    )


def get_mixture(
    mean_1: float,
    mean_2: float,
    log_sigma_1: float,
    log_sigma_2: float,
    mixture_ratio: float,
) -> BayesianConversionConfig:
    kwargs = {
        "mean_1": mean_1,
        "mean_2": mean_2,
        "log_sigma_1": log_sigma_1,
        "log_sigma_2": log_sigma_2,
        "mixture_ratio": mixture_ratio,
    }
    return BayesianConversionConfig(
        modules_to_replace={
            Linear: BayesianModuleConfig(
                module=BayesianLinear,
                priors={
                    "weight": ScaleMixturePrior(**kwargs),
                    "bias": ScaleMixturePrior(**kwargs),
                },
            ),
            Conv2d: BayesianModuleConfig(
                module=BayesianConv2d,
                priors={
                    "weight": ScaleMixturePrior(**kwargs),
                    "bias": ScaleMixturePrior(**kwargs),
                },
            ),
            BatchNorm2d: BayesianModuleConfig(
                module=BayesianNop,
                priors={},
            ),
        }
    )
