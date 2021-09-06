import torch
from pytorch_lightning import Trainer, Callback
from src.data.polynomial import PolynomialDataModule
from src.experiments.common import get_args
from src.inference import BayesianRegressor
from src.models.polynomial import PolynomialModel
from typing import List

class LogPosteriorSamples(Callback):

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):

        if pl_module.global_step < pl_module.burn_in:
            # Burn in sample
            return 

        if (pl_module.burn_in + pl_module.global_step) % pl_module.use_every != 0:
            # Thin sample
            return 

        for k, param in pl_module.model.named_parameters():
            for i, value in enumerate(param.flatten()):
                pl_module.log(f"weights/{k}.{i}", value)
   
def main():

    args = get_args(PolynomialModel, BayesianRegressor)

    torch.manual_seed(123)

    dm = PolynomialDataModule(args.batch_size, train_obs=1000, val_obs=100)
    sampler = args.sampler.from_argparse_args(args)
    model = PolynomialModel(torch.randn(4))
    inference = BayesianRegressor.from_argparse_args(args, model=model, sampler=sampler)

    trainer = Trainer.from_argparse_args(args, callbacks=[LogPosteriorSamples()])
    trainer.fit(inference, dm)

if __name__ == "__main__":

    main()
