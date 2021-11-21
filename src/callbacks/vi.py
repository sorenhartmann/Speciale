from pytorch_lightning import Callback

from src.inference.vi import VariationalModule


class LogVariationalParameterDistribution(Callback):
    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
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
