from pytorch_lightning import Callback, Trainer
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from src.data.mnist import MNISTDataModule
from src.experiments.common import (
    ExperimentHandler,
    FlatTensorBoardLogger,
)
import torch
from pathlib import Path
import argparse
import re
from src.inference.factory import inference_from_config

def infer_type(s):
    try:
        s = float(s)
        if s // 1 == s:
            return int(s)
        return s
    except ValueError:
        return s

def get_inference_args(
    default_model: str = "mlp_classifier",
    default_inference: str = "mcmc",
):

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default=default_model, type=str)
    parser.add_argument("--inference", default=default_inference, type=str)

    Trainer.add_argparse_args(parser)
    _, extra_args = parser.parse_known_args()
    component_args = [
        match.group(1)
        for x in extra_args
        if (match := re.fullmatch("--([a-z.]+)=.+", x))
    ]
    for arg in component_args:
        parser.add_argument(f"--{arg}", dest=arg.replace(".", "__"), default=None, type=infer_type)
    namespace = parser.parse_args()

    return namespace

from src.inference.base import InferenceModule, get_hyperparameters

def inference_from_argparse_args(args) -> InferenceModule:

    inference_spec = {
        "model": args.model,
        "inference": args.inference,
    }
    inference_spec.update({k.replace("__", "."): v for k, v in vars(args).items() if "__" in k})
    inference = inference_from_config(inference_spec)
    return inference

# Consider adding handler as argument
def experiment():

    torch.manual_seed(123)

    args = get_inference_args()

    dm = MNISTDataModule(500)
    inference = inference_from_argparse_args(args)
    inference.save_hyperparameters()

    callbacks = [ModelCheckpoint("./checkpoints")]
    logger = FlatTensorBoardLogger("./metrics")

    trainer = Trainer.from_argparse_args(args, logger=logger, callbacks=callbacks)
    trainer.fit(inference, dm)

if __name__ == "__main__":

    handler = ExperimentHandler(experiment)
    handler.run()
