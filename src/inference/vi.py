import torch
from typing import Callable, Optional
from torch.distributions import Normal

from src.inference.base import InferenceModule
from src.models.base import Model
from src.utils import ModuleAttributeHelper


from src.bayesian.core import (
    to_bayesian_model,
    BayesianConversionConfig,
    BayesianModuleConfig,
)
from src.bayesian.modules import BayesianModule, BayesianNop, BayesianLinear
from src.bayesian.priors import ScaleMixturePrior


from torch import nn
import torch
from torch.distributions import Normal

_INITIAL_RHO = -5
_DEFAULT_VI_PRIORS = BayesianConversionConfig(
    {
        nn.Linear: BayesianModuleConfig(
            module=BayesianLinear,
            priors={
                "weight": ScaleMixturePrior(),
                "bias": ScaleMixturePrior(),
            },
        ),
        nn.Conv2d: BayesianModuleConfig(
            module=ScaleMixturePrior,
            priors={
                "weight": ScaleMixturePrior(),
                "bias": ScaleMixturePrior(),
            },
        ),
        nn.BatchNorm2d: BayesianModuleConfig(
            module=BayesianNop,
            priors={},
        ),
    }
)


class VariationalModule(nn.Module):
    def __init__(self, bayesian_module: BayesianModule):

        super().__init__()

        self.bayesian_module = bayesian_module
        self.variational_parameters = nn.ModuleDict()

        parameter_names = [
            n for n, _ in bayesian_module.named_parameters(recurse=False)
        ]

        for name in parameter_names:

            data = getattr(bayesian_module, name).data
            del bayesian_module._parameters[name]

            self.variational_parameters[name] = nn.Module()
            self.variational_parameters[name].mu = nn.Parameter(data)
            self.variational_parameters[name].rho = nn.Parameter(
                torch.zeros_like(data) + _INITIAL_RHO
            )

    def sample_parameters(self, n_particles=1):

        self.log_v_post_ = 0
        self.sampled_parameters_ = {}
        for name, v_param in self.variational_parameters.items():

            expanded_shape = (n_particles,) + v_param.mu.shape
            sigma = v_param.rho.exp().log1p()
            eps = torch.rand(expanded_shape, device=sigma.device)
            sampled_parameters = eps * sigma + v_param.mu

            # sampled_parameters.requires_grad_()
            # sampled_parameters.retain_grad()
            # # Save for backward pass
            # with torch.no_grad():
            #     v_param.rho_gradient_mult_ = eps / (1 + torch.exp(-v_param.rho))

            # Save varitational log prob
            _sum_dims = tuple(range(1, len(expanded_shape)))
            self.log_v_post_ += (
                Normal(v_param.mu, sigma).log_prob(sampled_parameters).sum(_sum_dims)
            )

            self.sampled_parameters_[name] = sampled_parameters

        self.particle_idx = 0

    def forward(self, x):

        for name, sampled_parameter in self.sampled_parameters_.items():
            setattr(self.bayesian_module, name, sampled_parameter[self.particle_idx])
        self.particle_idx += 1
        return self.bayesian_module(x)

    # @torch.no_grad()
    # def update_gradients_(self):

    #     # Should be called after backward
    #     for name, v_param in self.variational_parameters.items():
    #         sampled_param = getattr(self.bayesian_module, name)
    #         v_param.mu.grad += sampled_param.grad.sum(0)
    #         v_param.rho.grad += (v_param.rho_gradient_mult_ * sampled_param.grad).sum(0)

    def log_v_post(self):
        return self.log_v_post_

    def log_prior(self):

        log_prior = 0
        for name in self.variational_parameters:
            _sum_dims = tuple(range(1, self.sampled_parameters_[name].dim()))
            log_prior += (
                self.bayesian_module.priors[name]
                .log_prob(self.sampled_parameters_[name])
                .sum(_sum_dims)
            )
        return log_prior


from src.utils import ModuleAttributeHelper


def to_variational_model_(model):
    def replace_submodules_(module: nn.Module):

        helper = ModuleAttributeHelper(module)
        for key, child in helper.keyed_children():
            if isinstance(child, BayesianNop):
                pass
            elif isinstance(child, BayesianModule):
                new_module = VariationalModule(child)
                helper[key] = new_module
            else:
                replace_submodules_(child)

    replace_submodules_(model)
    return model


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


class VariationalInference(InferenceModule):

    # TODO: specifiy prior
    def __init__(
        self,
        model: Model,
        lr: float = 1e-3,
        n_particles=2,
        prior_config=None,
        kl_weighting_scheme: Optional[KLWeightingScheme] = None,
    ):

        super().__init__()

        # self.automatic_optimization = False
        self.lr = lr
        self.n_particles = n_particles

        if kl_weighting_scheme is None:
            kl_weighting_scheme = ConstantKLWeight()

        self.kl_weighting_scheme = kl_weighting_scheme

        if prior_config is None:
            prior_config = _DEFAULT_VI_PRIORS

        model = to_bayesian_model(model, prior_config)
        self.model = to_variational_model_(model)

        self.train_metrics = torch.nn.ModuleDict(self.model.get_metrics())
        self.val_metrics = torch.nn.ModuleDict(self.model.get_metrics())

    def sample_parameters(self, n_particles):
        for module in self.variational_modules():
            module.sample_parameters(n_particles)

    def forward_particles(self, x):

        self.sample_parameters(self.n_particles)
        outputs = []
        for i in range(self.n_particles):
            outputs.append(self.model(x))

        return outputs

    def training_step(self, batch, batch_idx):

        x, y = batch

        output = torch.stack(self.forward_particles(x))

        obs_model = self.model.observation_model_gvn_output(output)
        log_lik = obs_model.log_prob(y).sum(-1)
        kl = self.log_v_post() - self.log_prior()
        kl_w = self.kl_weighting_scheme(batch_idx, len(self.trainer.train_dataloader))
        batch_elbo = (log_lik - kl * kl_w).mean(0)
        batch_elbo = batch_elbo

        preds = self.model.predict_gvn_output(output).mean(0)
        self.log("elbo/train", batch_elbo, on_step=False, on_epoch=True)
        self.log("kl/train", kl.mean())
        self.log("log_lik/train", log_lik.mean())
        for name, metric in self.train_metrics.items():
            self.log(f"{name}/train", metric(preds, y), on_epoch=True, on_step=False)

        return -batch_elbo

    def validation_step(self, batch, batch_idx):

        x, y = batch
        preds = torch.stack(self.forward_particles(x)).softmax(-1).mean(0)
        for name, metric in self.val_metrics.items():
            self.log(f"{name}/val", metric(preds, y), prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.SGD(self.variational_parameters(), lr=self.lr)

    def variational_modules(self):
        yield from filter(lambda x: isinstance(x, VariationalModule), self.modules())

    def variational_parameters(self):
        for module in self.variational_modules():
            yield from module.parameters()

    def log_prior(self):
        return sum(module.log_prior() for module in self.variational_modules())

    def log_v_post(self):
        return sum(module.log_v_post() for module in self.variational_modules())
