

class LogVarianceEstimates(Callback):

    interbatch_variance_folder = Path("variance_interbatch")
    variance_estimated_folder = Path("variance_estimated")

    def __init__(self, n_gradients=1000, logs_per_epoch=10):

        self.n_gradients = n_gradients
        self.logs_per_epoch = logs_per_epoch

    def on_init_start(self, trainer) -> None:

        self.interbatch_variance_folder.mkdir()
        self.variance_estimated_folder.mkdir()

    def _get_estimate(self, estimator):
        if type(estimator) is VarianceEstimatorWrapper:
            return estimator.wrapped.estimate()
        else:
            return estimator.estimate()

    def on_fit_start(self, trainer, pl_module) -> None:

        with torch.random.fork_rng():
            torch.manual_seed(123)
            self.log_idx, _ = torch.sort(
                torch.randperm(pl_module.posterior.shape[0])[: self.n_gradients]
            )
        torch.save(self.log_idx, "log_idx.pt")

    @cache
    def sample_steps(self, trainer):
        num_batches = len(trainer.train_dataloader)
        steps_between_samples = num_batches / self.logs_per_epoch
        return [
            int(i * steps_between_samples) - 1
            for i in range(1, self.logs_per_epoch + 1)
        ]

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, unused
    ) -> None:

        if batch_idx in self.sample_steps(trainer):
            self.log_gradient_estimate(trainer, pl_module, batch_idx)

    def log_gradient_estimate(self, trainer, pl_module, batch_idx):

        with torch.random.fork_rng():

            wf_estimator = WelfordEstimator()
            for batch in trainer.train_dataloader:
                x, y = batch
                sampling_fraction = len(x) / len(trainer.train_dataloader.dataset)
                with pl_module.posterior.observe(x, y, sampling_fraction):
                    wf_estimator.update(pl_module.posterior.grad_prop_log_p())

        variance = wf_estimator.estimate()
        estimate = self._get_estimate(pl_module.sampler.variance_estimator)

        variance, estimate = torch.broadcast_tensors(variance, estimate)

        torch.save(
            variance[self.log_idx],
            self.interbatch_variance_folder / f"{trainer.global_step:06}.pt",
        )
        torch.save(
            estimate[self.log_idx],
            self.variance_estimated_folder / f"{trainer.global_step:06}.pt",
        )

