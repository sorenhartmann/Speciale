from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities.types import STEP_OUTPUT

from src.inference.base import BATCH_IN
from src.inference.vi import VariationalModule


class LogVariationalParameterDistribution(Callback):
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: BATCH_IN,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:

        if batch_idx == len(trainer.train_dataloader) - 1:

            for module_name, module in pl_module.named_modules():

                if not isinstance(module, VariationalModule):
                    continue

                for param_name, param in module.named_parameters():

                    pl_module.logger.experiment.add_histogram(
                        f"{module_name}[{param_name}]/value",
                        param,
                        pl_module.current_epoch,
                    )
                    pl_module.logger.experiment.add_histogram(
                        f"{module_name}[{param_name}]/grad",
                        param.grad,
                        pl_module.current_epoch,
                    )
