from pytorch_lightning import Callback


class LogVariationalParameterDistribution(Callback):
    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ) -> None:

        if batch_idx == len(trainer.train_dataloader) - 1:
            pl_module.logger.experiment.add_histogram(
                "mu/value", pl_module.mu, pl_module.current_epoch
            )
            pl_module.logger.experiment.add_histogram(
                "mu/grad", pl_module.mu.grad, pl_module.current_epoch
            )
            pl_module.logger.experiment.add_histogram(
                "rho/value", pl_module.rho, pl_module.current_epoch
            )
            pl_module.logger.experiment.add_histogram(
                "rho/grad", pl_module.rho.grad, pl_module.current_epoch
            )

