
import torch
from pytorch_lightning import Trainer
from src.data.polynomial import PolynomialDataModule
from src.experiments.common import get_args
from src.inference import BayesianRegressor
from src.models.polynomial import PolynomialModel


def main():

    args = get_args(PolynomialModel, BayesianRegressor)

    torch.manual_seed(123)

    dm = PolynomialDataModule(args.batch_size, train_obs=100, val_obs=20)
    sampler = args.sampler.from_argparse_args(args)
    model = PolynomialModel(torch.randn(4))
    inference = BayesianRegressor.from_argparse_args(
        args, model=model, sampler=sampler, log_samples=True
    )

    # Burn in
    trainer = Trainer.from_argparse_args(
        args,
        # logger=WandbLogger(project="bayesian-deep-learning")
    )
    trainer.fit(inference, dm)

    pass


if __name__ == "__main__":

    main()
