import torch
from pytorch_lightning import Trainer
from torch.distributions import Normal

from src.data.mnist import MNISTDataModule
from src.inference.base import InferenceModule
from src.inference.probabilistic import (NormalMixturePrior, PriorSpec,
                                         as_probabilistic_model)
from src.models.base import Model
from src.models.mlp import MLPClassifier
from src.utils import ParameterView


def bufferize_parameters_(module):
    parameter_names = [n for n, _ in module.named_parameters(recurse=False)]
    for name in parameter_names:
        buffer = getattr(module, name).data
        del module._parameters[name]
        module.register_buffer(name, buffer)


class VariationalInference(InferenceModule):

    # TODO: specifiy prior
    def __init__(
        self,
        model: Model,
        lr: float = 1e-3,
        n_samples=10,
        prior_spec=None,
        initial_rho=-2,
    ):

        super().__init__()

        # self.automatic_optimization = False
        self.lr = lr
        self.n_samples = n_samples

        if prior_spec is None:
            prior_spec = PriorSpec(NormalMixturePrior())

        self.model = as_probabilistic_model(model, prior_spec)
        parameter_names = [n for n, _ in self.model.named_parameters()]
        for module in self.model.modules():
            bufferize_parameters_(module)

        self.view = ParameterView(self.model, buffers=parameter_names)
        self.register_parameter(
            "rho", torch.nn.Parameter(initial_rho * torch.ones(self.view.n_params))
        )
        self.register_parameter(
            "mu", torch.nn.Parameter(torch.zeros(self.view.n_params))
        )

        self.train_metrics = self.model.get_metrics()
        self.val_metrics = self.model.get_metrics()

    def training_step(self, batch, batch_idx):

        x, y = batch

        sigma = torch.log(1 + self.rho.exp())
        eps = torch.randn_like(self.mu)
        w = self.mu + sigma * eps

        w.requires_grad_(True)
        w.retain_grad()

        self.view[:] = w

        kl = Normal(self.mu, sigma).log_prob(w).sum() - self.model.log_prior()

        output = self.model(x)
        obs_model = self.model.observation_model_gvn_output(output)
        log_lik = obs_model.log_prob(y).sum()

        M = len(self.trainer.train_dataloader)

        elbo = log_lik - kl / M

        # Save references for grad
        self._eps = eps
        self._w = w

        self.log("elbo/train", elbo)
        self.log("kl/train", kl)
        self.log("log_lik/train", log_lik)
        for name, metric in self.train_metrics.items():
            self.log(f"{name}/train", metric(output, y), on_epoch=True, on_step=False)

        return -elbo

    def on_after_backward(self) -> None:

        with torch.no_grad():
            self.mu.grad.add_(self._w.grad)
            self.rho.grad.add_(self._w.grad * (self._eps / (1 + torch.exp(-self.rho))))

        del self._w
        del self._eps

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx) -> None:
        if batch_idx == len(self.trainer.train_dataloader) - 1:
            self.logger.experiment.add_histogram(
                "mu/value", self.mu, self.current_epoch
            )
            self.logger.experiment.add_histogram(
                "mu/grad", self.mu.grad, self.current_epoch
            )
            self.logger.experiment.add_histogram(
                "rho/value", self.rho, self.current_epoch
            )
            self.logger.experiment.add_histogram(
                "rho/grad", self.rho.grad, self.current_epoch
            )

    def on_validation_epoch_start(self) -> None:

        self.w_samples = torch.randn((self.n_samples,) + self.mu.shape)
        self.w_samples.mul_(torch.log(1 + self.rho.exp()))
        self.w_samples.add_(self.mu)

    def validation_step(self, batch, batch_idx):

        x, y = batch

        prediction = 0
        for w in self.w_samples:
            self.view[:] = w
            prediction += self.model.predict(x)

        prediction /= self.n_samples

        # if batch_idx == 0:
        #     print("ayy")

        # self.log("max_p/val", output., on_epoch=True, on_step=False)

        for name, metric in self.val_metrics.items():
            self.log(f"{name}/val", metric(prediction, y), prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        del self.w_samples

    def configure_optimizers(self):

        optimizer = torch.optim.SGD((self.mu, self.rho), lr=self.lr)
        return optimizer
