from torch.nn import Linear, Conv2d, BatchNorm2d
from src.bayesian.modules import BayesianLinear, BayesianNop, BayesianConv2d
from src.bayesian.priors import NormalPrior, ScaleMixturePrior
from src.bayesian.core import BayesianModuleConfig, BayesianConversionConfig


def get_normal(
    mean=0.,
    precision=1.
):
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
    mean_1,
    mean_2,
    log_sigma_1,
    log_sigma_2,
    mixture_ratio,
):
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
