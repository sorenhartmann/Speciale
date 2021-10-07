from src.inference.base import InferenceModule
from src.inference.probabilistic import to_probabilistic_model_
from src.utils import HPARAM, Component, ParameterView_, register_component
from src.models.base import Model
import torch
from torch.distributions import Normal

from src.models.mlp import MLPClassifier
from pytorch_lightning import Trainer
from src.data.mnist import MNISTDataModule

def bufferize_parameters_(module):
    parameter_names = [n for n, _ in module.named_parameters(recurse=False)]
    for name in parameter_names:
        buffer = getattr(module, name).data
        del module._parameters[name]
        module.register_buffer(name, buffer)

@register_component("vi")
class VariationalInference(InferenceModule):

    model : HPARAM[Component]
    lr : HPARAM[float]
    n_samples : HPARAM[int]

    # TODO: specifiy prior
    def __init__(self, model: Model, lr: float = 1e-3, n_samples=10, prior=None):

        super().__init__()

        # self.automatic_optimization = False
        self.lr = lr
        self.n_samples = n_samples

        self.model = model
        to_probabilistic_model_(self.model)
        parameter_names = [n for n, _ in model.named_parameters()]
        for module in self.model.modules():
            bufferize_parameters_(module)

        self.view = ParameterView_(self.model, buffers=parameter_names)
        self.register_parameter(
            "rho", torch.nn.Parameter(torch.zeros(self.view.n_params))
        )
        self.register_parameter(
            "mu", torch.nn.Parameter(torch.zeros(self.view.n_params))
        )

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
        log_lik = self.model.log_likelihood(x, y).sum()

        elbo = kl / len(self.trainer.train_dataloader) - log_lik

        self._eps = eps  # Save reference for grad
        self._w = w  # Save reference for grad

        self.log("elbo/train", elbo, prog_bar=True, on_epoch=True, on_step=False)

        return elbo

    def on_after_backward(self) -> None:

        with torch.no_grad():
            self.mu.grad.add_(self._w.grad)
            self.rho.grad.add_(self._w.grad * (self._eps / (1 + torch.exp(-self.rho))))

        del self._w
        del self._eps

    def on_validation_epoch_start(self) -> None:

        self.w_samples = torch.randn((self.n_samples, ) + self.mu.shape)
        self.w_samples.mul_(torch.log(1 + self.rho.exp()))
        self.w_samples.add_(self.mu)

    def validation_step(self, batch, batch_idx):

        x, y = batch

        output = 0
        for w in self.w_samples:
            self.view[:] = w
            output += self.model.predict(x)
        
        output /= self.n_samples

        for name, metric in self.val_metrics.items():
            self.log(f"{name}/val", metric(output, y))

    def on_validation_epoch_end(self) -> None:
        del self.w_samples

    def configure_optimizers(self):

        optimizer = torch.optim.SGD((self.mu, self.rho), lr=self.lr)
        return optimizer


if __name__ == "__main__":

    model = MLPClassifier()
    inference = VariationalInference(model)
    datamodule = MNISTDataModule(128)

    Trainer().fit(inference, datamodule)
