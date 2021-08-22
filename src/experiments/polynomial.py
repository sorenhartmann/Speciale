from argparse import ArgumentParser
import pytorch_lightning

import torch
from pytorch_lightning import Trainer
from src.data.polynomial import PolynomialDataModule
from src.inference import BayesianInference
from src.models.polynomial import PolynomialModel
from src.samplers import GetSampler


def get_args():

    parser = ArgumentParser()

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
    pass  # Polynomial model has no args

    # Inference specific args
    parser = BayesianInference.add_argparse_args(parser)

    # Training specific args
    parser = Trainer.add_argparse_args(parser)

    # Parse and return
    args = parser.parse_args()
    return args


def main():

    args = get_args()

    torch.manual_seed(123)

    dm = PolynomialDataModule(args.batch_size, train_obs=1000, val_obs=1000)
    sampler = args.sampler.from_argparse_args(args)
    model = PolynomialModel(torch.randn(4))
    inference = BayesianInference.from_argparse_args(
        args, model=model, sampler=sampler, log_samples=True
    )

    # Burn in
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(inference, dm)

    pass


if __name__ == "__main__":

    main()
