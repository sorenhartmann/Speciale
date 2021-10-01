from src.inference.base import InferenceModule, to_probabilistic_model_
from src.utils import ParameterView
from src.models.base import Model
import torch
from torch.distributions import Normal


class VariationalInference(InferenceModule):
    def __init__(self, model: Model, lr: float = 1e-3):

        super().__init__()

        self.model = model
        to_probabilistic_model_(self.model)

        self.lr = lr

        self.parameter_view = ParameterView(self.model)
        self.register_buffer(
            "rho", torch.zeros(self.parameter_view.n_params, requires_grad=True)
        )
        self.register_buffer(
            "mu", torch.zeros(self.parameter_view.n_params, requires_grad=True)
        )

        self.train_metrics = self.model.get_metrics()
        self.val_metrics = self.model.get_metrics()

        # TODO: Refactor later, probably in a factory?
        self.save_hyperparameters({"inference_type": "VI"})

    def training_step(self, batch, batch_idx):

        x, y = batch
        self.parameter_view.apply(lambda x: setattr(x, "requires_grad", False))

        # Resample parameters
        sigma = torch.log(1 + self.rho.exp())
        eps = torch.randn_like(sigma)
        w = self.mu + sigma * eps
        self.parameter_view[:] = w

        # Retain weight/bias grads
        self.parameter_view.apply(lambda x: x.retain_grad())

        elbo = (
            Normal(self.mu, sigma).log_prob(w).sum()
            - self.model.log_likelihood(x, y).sum()
            - self.model.log_prior()
        )

        return elbo

    def configure_optimizers(self):
        optimizer = torch.optim.Adam((self.mu, self.rho), lr=self.lr)
        return optimizer
