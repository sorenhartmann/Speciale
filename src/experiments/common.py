from src.experiments.synthetic import HamiltonianNoMH
from src.samplers import Hamiltonian, MetropolisHastings, StochasticGradientHamiltonian
import argparse
from typing import Type
from src.modules import BayesianModel
from src.inference import BayesianRegressor
from pytorch_lightning import Trainer

samplers = {
    sampler_cls.tag: sampler_cls
    for sampler_cls in [
        MetropolisHastings,
        Hamiltonian,
        StochasticGradientHamiltonian,
        HamiltonianNoMH,
    ]
}

class GetSampler(argparse.Action):

    default = Hamiltonian

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, samplers[values])


def get_args(model_cls: Type[BayesianModel], inference_cls: Type[BayesianRegressor]):

    parser = argparse.ArgumentParser()

    # Batch size (for batched inference only)
    parser.add_argument("--batch_size", type=int, default=8)

    # Sampler and sampler args
    parser.add_argument(
        "--sampler",
        action=GetSampler,
        default=GetSampler.default,
        choices=GetSampler.samplers.keys(),
    )
    known_args, _ = parser.parse_known_args()
    parser = known_args.sampler.add_argparse_args(parser)

    # Model specific args
    parser = model_cls.add_argparse_args(parser)

    # Inference specific args
    parser = inference_cls.add_argparse_args(parser)

    # Training specific args
    parser = Trainer.add_argparse_args(parser)

    # Parse and return
    args = parser.parse_args()
    return args
