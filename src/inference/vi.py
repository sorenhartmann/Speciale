from typing import Iterator, List, Optional, cast

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor, nn
from torch.distributions import Normal
from torch.nn import Parameter

from src.bayesian.core import (
    BayesianConversionConfig,
    BayesianModuleConfig,
    to_bayesian_model,
)
from src.bayesian.modules import (
    BayesianConv2d,
    BayesianLinear,
    BayesianModule,
    BayesianNop,
)
from src.bayesian.priors import ScaleMixturePrior
from src.inference.base import BATCH_IN, InferenceModule
from src.models.base import ErrorRate, Model
from src.utils import ModuleAttributeHelper

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
            module=BayesianConv2d,
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
    def __init__(
        self,
        bayesian_module: BayesianModule,
        initial_rho: float = -5,
    ) -> None:

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
                torch.zeros_like(data) + initial_rho
            )

    def sample_parameters(
        self,
        n_particles: int = 1,
        _save_for_backward: bool = False,
    ) -> None:

        self.log_v_post_ = torch.tensor(0)
        self.sampled_parameters_ = {}
        for name, v_param in self.variational_parameters.items():

            expanded_shape = (n_particles,) + v_param.mu.shape
            sigma = v_param.rho.exp().log1p()
            eps = torch.randn(expanded_shape, device=sigma.device)
            sampled_parameters = eps * sigma + v_param.mu

            # UNSURE IF NEEDED
            if _save_for_backward:
                sampled_parameters.requires_grad_()
                sampled_parameters.retain_grad()
                # Save for backward pass
                with torch.no_grad():
                    v_param.rho_gradient_mult_ = eps / (1 + torch.exp(-v_param.rho))

            # Save varitational log prob
            _sum_dims = tuple(range(1, len(expanded_shape)))
            self.log_v_post_ = self.log_v_post_ + cast(
                Tensor,
                Normal(v_param.mu, sigma).log_prob(sampled_parameters).sum(_sum_dims),
            )

            self.sampled_parameters_[name] = sampled_parameters

        self.particle_idx = 0

    def forward(self, x: Tensor) -> Tensor:

        for name, sampled_parameter in self.sampled_parameters_.items():
            setattr(self.bayesian_module, name, sampled_parameter[self.particle_idx])
        self.particle_idx += 1
        return self.bayesian_module(x)

    # UNSURE IF NEEDED
    @torch.no_grad()
    def update_gradients_(self) -> None:

        # Should be called after backward
        for name, v_param in self.variational_parameters.items():
            sampled_param = self.sampled_parameters_[name]
            v_param.mu.grad += sampled_param.grad.sum(0)
            v_param.rho.grad += (v_param.rho_gradient_mult_ * sampled_param.grad).sum(0)

    def log_v_post(self) -> Tensor:
        return self.log_v_post_

    def log_prior(self) -> Tensor:

        log_prior = torch.tensor(0.0)
        for name in self.variational_parameters:
            _sum_dims = tuple(range(1, self.sampled_parameters_[name].dim()))
            log_prior = log_prior + (
                self.bayesian_module.priors[name]
                .log_prob(self.sampled_parameters_[name])
                .sum(_sum_dims)
            )
        return log_prior


from src.utils import ModuleAttributeHelper


def to_variational_model_(model: Model, initial_rho: float = -5) -> Model:
    def replace_submodules_(module: nn.Module) -> Model:
        helper = ModuleAttributeHelper(module)
        for key, child in helper.keyed_children():
            if isinstance(child, BayesianNop):
                pass
            elif isinstance(child, BayesianModule):
                new_module = VariationalModule(child, initial_rho=initial_rho)
                helper[key] = new_module
            else:
                replace_submodules_(child)

    replace_submodules_(model)
    return model


class KLWeightingScheme:
    @staticmethod
    def get_weight(batch_idx: int, M: int) -> float:
        ...


class ConstantKLWeight(KLWeightingScheme):
    @staticmethod
    def get_weight(batch_idx: int, M: int) -> float:
        return 1 / M


class ExponentialKLWeight(KLWeightingScheme):
    @staticmethod
    def get_weight(batch_idx: int, M: int) -> float:
        weight = 2 ** (M - (batch_idx + 1)) / (2 ** M - 1)
        if weight < 1e-8:
            weight = 0.0
        return weight


class VariationalInference(InferenceModule):
    def __init__(
        self,
        model: Model,
        lr: float = 1e-3,
        n_particles: int = 2,
        prior_config: BayesianConversionConfig = None,
        kl_weighting_scheme: Optional[KLWeightingScheme] = None,
        initial_rho: float = -5,
        _adjust_gradients: bool = False,
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
        self.model = to_variational_model_(model, initial_rho=initial_rho)

        self.train_metrics = torch.nn.ModuleDict(self.model.get_metrics())
        self.val_metrics = torch.nn.ModuleDict(self.model.get_metrics())

        self._adjust_gradients = _adjust_gradients

    def sample_parameters(self, n_particles: int) -> None:
        for module in self.variational_modules():
            module.sample_parameters(
                n_particles, _save_for_backward=self._adjust_gradients
            )

    def forward_particles(self, x: Tensor) -> List[Tensor]:

        self.sample_parameters(self.n_particles)
        outputs = []
        for _ in range(self.n_particles):
            outputs.append(self.model(x))

        return outputs

    def training_step(self, batch: BATCH_IN, batch_idx: int) -> Optional[STEP_OUTPUT]:  # type: ignore

        x, y = batch

        trainer = cast(Trainer, self.trainer)

        output = torch.stack(self.forward_particles(x))

        obs_model = self.model.observation_model_gvn_output(output)
        log_lik = obs_model.log_prob(y).squeeze().sum(-1)
        kl = self.log_v_post() - self.log_prior()
        kl_w = self.kl_weighting_scheme.get_weight(
            batch_idx, len(trainer.train_dataloader)
        )
        batch_elbo = (log_lik - kl * kl_w).mean(0)

        preds = self.model.predict_gvn_output(output).mean(0)
        self.log("elbo/train", batch_elbo, on_step=False, on_epoch=True)
        self.log("kl/train", kl.mean())
        self.log("log_lik/train", log_lik.mean())
        for name, metric in self.train_metrics.items():
            self.log(f"{name}/train", metric(preds, y), on_epoch=True, on_step=False)

        return -batch_elbo

    def validation_step(self, batch: BATCH_IN, batch_idx: int) -> Optional[STEP_OUTPUT]:  # type: ignore

        x, y = batch
        preds = torch.stack(self.forward_particles(x)).softmax(-1).mean(0)
        for name, metric in self.val_metrics.items():
            self.log(f"{name}/val", metric(preds, y), prog_bar=True)

    def on_test_epoch_start(self) -> None:
        self.test_metric = ErrorRate().to(device=self.device)

    def test_step(self, batch: BATCH_IN, batch_idx: int) -> Optional[STEP_OUTPUT]:  # type: ignore

        x, y = batch
        preds = torch.stack(self.forward_particles(x)).softmax(-1).mean(0)
        self.log(f"err/test", self.test_metric(preds, y), prog_bar=True)

        return {"predictions": preds}

    def on_after_backward(self) -> None:
        if self._adjust_gradients:
            for module in self.variational_modules():
                module.update_gradients_()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.variational_parameters(), lr=self.lr)

    def variational_modules(self) -> Iterator[VariationalModule]:
        for module in self.modules():
            if isinstance(module, VariationalModule):
                yield module

    def variational_parameters(self) -> Iterator[Parameter]:
        for module in self.variational_modules():
            yield from module.parameters()

    def log_prior(self) -> Tensor:
        return sum(
            (module.log_prior() for module in self.variational_modules()),
            torch.tensor(0),
        )

    def log_v_post(self) -> Tensor:
        return sum(
            (module.log_v_post() for module in self.variational_modules()),
            torch.tensor(0),
        )
