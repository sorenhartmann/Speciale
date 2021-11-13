import torch
from typing import Callable, Optional, OrderedDict
from pytorch_lightning import Trainer
from torch import distributions
from torch.distributions import Normal
from collections import OrderedDict
from src.data.mnist import MNISTDataModule
from src.inference.base import InferenceModule
from src.inference.probabilistic import (
    NormalMixturePrior,
    PriorSpec,
    as_probabilistic_model,
)
from src.models.base import Model
from src.models.mlp import MLPClassifier
from src.utils import ParameterView


def bufferize_parameters_(module):
    parameter_names = [n for n, _ in module.named_parameters(recurse=False)]
    for name in parameter_names:
        buffer = getattr(module, name).data
        del module._parameters[name]
        module.register_buffer(name, buffer)


class KLWeightingScheme(Callable):
    ...


class ConstantKLWeight(KLWeightingScheme):
    @staticmethod
    def __call__(batch_idx, M):
        return 1 / M


class ExponentialKLWeight(KLWeightingScheme):
    @staticmethod
    def __call__(batch_idx, M):
        weight = 2 ** (M - (batch_idx + 1)) / (2 ** M - 1)
        if weight < 1e-8:
            weight = 0.0
        return weight


from torch.nn import ModuleDict, Module


class VariationalInference(InferenceModule):

    # TODO: specifiy prior
    def __init__(
        self,
        model: Model,
        lr: float = 1e-3,
        n_samples=10,
        prior_spec=None,
        initial_rho=-5,
        kl_weighting_scheme: Optional[KLWeightingScheme] = None,
    ):

        super().__init__()

        # self.automatic_optimization = False
        self.lr = lr
        self.n_samples = n_samples

        if kl_weighting_scheme is None:
            kl_weighting_scheme = ConstantKLWeight()

        self.kl_weighting_scheme = kl_weighting_scheme

        if prior_spec is None:
            prior_spec = PriorSpec(NormalMixturePrior())

        self.model = as_probabilistic_model(model, prior_spec)

        self.v_params = ModuleDict()
        for name, parameter in self.model.named_parameters():
            v_name = name.replace(".", "/")
            shape = parameter.shape
            self.v_params[v_name] = Module()
            self.v_params[v_name].register_parameter("mu", torch.zeros(shape))
            self.v_params[v_name].register_parameter("rho", initial_rho + torch.zeros(shape))

        self.train_metrics = torch.nn.ModuleDict(self.model.get_metrics())
        self.val_metrics = torch.nn.ModuleDict(self.model.get_metrics())

    def training_step(self, batch, batch_idx):

        x, y = batch

        sample = self.get_parameter_sample(cache=True)
        self.model.load_state_dict(sample, strict=False)

        kl = sum(v_param.log_prob_ for v_param in self.v_params.children())
        kl -= self.model.log_prior()

        output = self.model(x)
        obs_model = self.model.observation_model_gvn_output(output)
        log_lik = obs_model.log_prob(y).sum()

        self.kl_w_ = self.kl_weighting_scheme(batch_idx, len(self.trainer.train_dataloader))
        batch_elbo = log_lik - kl * self.kl_w_

        self.log("elbo/train", batch_elbo, on_step=False, on_epoch=True)
        self.log("kl/train", kl)
        self.log("log_lik/train", log_lik)
        for name, metric in self.train_metrics.items():
            self.log(f"{name}/train", metric(output, y), on_epoch=True, on_step=False)

        return -batch_elbo

    def normal_log_prob_partial_derivatives(self, mu, rho, parameter_value):

        param_mu = torch.nn.Parameter(mu)
        param_rho = torch.nn.Parameter(rho)
        distribution = Normal(param_mu, torch.log1p(param_rho.exp()))
        distribution.log_prob(parameter_value).sum().backward()
        return param_mu.grad, param_rho.grad

    def on_after_backward(self) -> None:

        for (name, parameter), (v_param) in zip(
            self.model.named_parameters(), self.v_params.children()
        ):

            partial_mu, partial_rho = self.normal_log_prob_partial_derivatives(
                v_param.mu, v_param.rho, parameter.detach()
            )

            v_param.mu.grad = parameter.grad
            v_param.rho.grad = parameter.grad



            self.mu.grad.add_(self._w.grad)
            self.rho.grad.add_(
                self._w.grad * (self._eps / torch.log1p(torch.exp(-self.rho)))
            )

        del self._w
        del self._eps

    def get_parameter_sample(self, cache=False):

        sample = {}
        for (name, parameter), (v_param) in zip(
            self.model.named_parameters(), self.v_params.children()
        ):
            eps = torch.randn(parameter.shape)
            sigma = torch.log1p(v_param.rho.exp())
            sample[name] = v_param.mu + eps * sigma

            if cache:
                v_param.eps_ = eps
                v_param.sigma_ = sigma
                v_param.log_prob_ = (
                    Normal(v_param.mu, sigma).log_prob(sample[name]).sum()
                )

        return sample

    def on_validation_epoch_start(self) -> None:
        self.samples_ = [self.get_parameter_sample() for i in range(self.n_samples)]

    def validation_step(self, batch, batch_idx):

        x, y = batch

        prediction = 0
        for sample in self.samples_:
            self.model.load_state_dict(sample, strict=False)
            prediction += self.model.predict(x)

        prediction /= self.n_samples

        for name, metric in self.val_metrics.items():
            self.log(f"{name}/val", metric(prediction, y), prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        del self.samples_

    def configure_optimizers(self):

        optimizer = torch.optim.SGD(self.v_params.buffers(), lr=self.lr)
        return optimizer
