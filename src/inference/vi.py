import torch
from pytorch_lightning import Trainer
from torch.distributions import Normal

from src.data.mnist import MNISTDataModule
from src.inference.base import InferenceModule
from src.inference.probabilistic import as_probabilistic_model
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
    def __init__(self, model: Model, lr: float = 1e-3, n_samples=10, prior=None):

        super().__init__()

        # self.automatic_optimization = False
        self.lr = lr
        self.n_samples = n_samples

        self.model = as_probabilistic_model(model)
        parameter_names = [n for n, _ in self.model.named_parameters()]
        for module in self.model.modules():
            bufferize_parameters_(module)

        self.view = ParameterView(self.model, buffers=parameter_names)
        self.register_parameter(
            "rho", torch.nn.Parameter(torch.zeros(self.view.n_params))
        )
        self.register_parameter(
            "mu", torch.nn.Parameter(torch.zeros(self.view.n_params))
        )

        self.train_metrics = self.model.get_metrics()
        self.val_metrics = self.model.get_metrics()

        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):

        x, y = batch

        sigma = torch.log(1 + self.rho.exp())
        eps = torch.randn_like(self.mu)
        w = self.mu + sigma * eps

        w.requires_grad_(True)
        w.retain_grad()

        self.view[:] = w

        N = len(self.trainer.train_dataloader.dataset)

        kl = (Normal(self.mu, sigma).log_prob(w).sum() - self.model.log_prior()) / N
        # kl = 0
        
        output = self.model(x)
        obs_model = self.model.observation_model_gvn_output(output)
        log_lik = obs_model.log_prob(y).mean()

        elbo = log_lik - kl 

        self._eps = eps  # Save reference for grad
        self._w = w  # Save reference for grad

        self.log("elbo/train", elbo, prog_bar=True, on_epoch=True, on_step=False)
        self.log("kl/train", kl, on_epoch=True, on_step=False)
        self.log("log_lik/train", log_lik, on_epoch=True, on_step=False)
        for name, metric in self.train_metrics.items():
            self.log(f"{name}/train", metric(output, y), on_epoch=True, on_step=False)


        self.zero_grad()
        (-elbo).backward()
        
        with torch.no_grad():
            self.mu.grad.add_(w.grad)
            self.rho.grad.add_(w.grad * (eps / (1 + torch.exp(-self.rho))))

            self.mu.sub_(self.mu.grad * self.lr)
            self.rho.sub_(self.rho.grad * self.lr)

    def on_validation_epoch_start(self) -> None:

        self.w_samples = torch.randn((self.n_samples, ) + self.mu.shape)
        self.w_samples.mul_(torch.log(1 + self.rho.exp()))
        self.w_samples.add_(self.mu)

    def validation_step(self, batch, batch_idx):

        x, y = batch

        prediction = 0
        for w in self.w_samples:
            self.view[:] = w
            prediction += self.model.predict(x)
        
        prediction /= self.n_samples
        # self.log("max_p/val", output., on_epoch=True, on_step=False)

        for name, metric in self.val_metrics.items():
            self.log(f"{name}/val", metric(prediction, y))


    def on_validation_epoch_end(self) -> None:
        del self.w_samples

    def configure_optimizers(self):

        # optimizer = torch.optim.SGD((self.mu, self.rho), lr=self.lr)
        return None

