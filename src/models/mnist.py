from src.utils import HPARAM, HyperparameterMixin
import torch
from torch import nn
from torch._C import Module

from src.data.mnist import MNISTDataModule
from src.samplers import Hamiltonian, Sampler, StochasticGradientHamiltonian
from src.modules import BayesianLinear, BayesianLinearKnownPrecision, BayesianMixin, BayesianModel
from tqdm import trange
from torch.distributions import Gamma
import math
import pytorch_lightning as pl
import torchmetrics

class MNISTModel(BayesianModel):

    def __init__(
        self, in_features=784, out_features=10, hidden_layers=[100], alpha=1.0, beta=1.0
    ):
        super().__init__()

        layers = []
        in_size = in_features
        for hidden_size in hidden_layers:
            out_size = hidden_size
            layers.append(
                BayesianLinearKnownPrecision(in_size, out_size).setup_prior(1.)
            )
            layers.append(nn.Sigmoid())
            in_size = out_size

        layers.append(
            BayesianLinearKnownPrecision(in_size, out_features).setup_prior(1.)
        )

        self.ffnn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        x = x.flatten(-2, -1)
        return self.ffnn(x)

    def observation_model(self, x: torch.FloatTensor):
        """Returns log p(y | x, theta)"""
        logits = self.forward(x)
        return torch.distributions.Categorical(logits=logits)

class BayesianClassifier(pl.LightningModule, HyperparameterMixin):

    burn_in: HPARAM[int]
    keep_samples: HPARAM[int]
    use_every: HPARAM[int]

    def __init__(
        self,
        model: BayesianModel,
        sampler: Sampler,
        burn_in=100,
        keep_samples=50,
        use_every=50,
    ):

        super().__init__()

        self.burn_in = burn_in
        self.keep_samples = keep_samples
        self.use_every = use_every

        self.model = model
        self.sampler = sampler

        self.automatic_optimization = False

        self.save_hyperparameters(self.get_hparams())
        self.save_hyperparameters(self.model.get_hparams())
        self.save_hyperparameters({"sampler": self.sampler.tag})
        self.save_hyperparameters(self.sampler.get_hparams())

        self.samples_ = []

    def configure_optimizers(self):
        return None

    def setup(self, stage) -> None:

        self.sampler.setup(self.model)
        if not self.sampler.is_batched:
            self.trainer.datamodule.batch_size = None

    def training_step(self, batch, batch_idx):

        x, y = batch
        sample = self.sampler.next_sample(x, y)

        if self.global_step < self.burn_in:
            # Burn in sample
            return 

        if (self.burn_in + self.global_step) % self.use_every != 0:
            # Thin sample
            return None

        if len(self.samples_) == self.keep_samples:
            # Discard oldest sample
            del self.samples_[0]

        self.samples_.append(sample)


    def validation_step(self, batch, batch_idx):

        if (
            len(self.samples_) == 0
            or (self.burn_in + self.global_step) % self.use_every != 0
        ):
            return

        with torch.no_grad():

            x, y = batch
            pred_samples = []
            for sample in self.samples_:
                self.model.flat_params = sample
                pred_samples.append(self.model.forward(x))

            y_hat = torch.stack(pred_samples).mean(0)

            self.log("loss/accuracy_mse", torchmetrics.functional.accuracy(y_hat, y))

def main():


    torch.manual_seed(123)

    dm = MNISTDataModule(500)
    model = MNISTModel()

    sampler = StochasticGradientHamiltonian(alpha=0.01, beta=0, eta=2e-6, n_steps=1)
    sampler.setup_sampler(model, train_data)

    for i in trange(800):

        samples = []
        for _ in range(100):
            samples.append(sampler.next_sample())
        
        
        # Update gamma
        with torch.no_grad():

            new_precisions = {}

            for name, param in model.named_parameters():
                param_samples = torch.stack([x[name] for x in samples])
                
                # Iterateively or with prior?
                precision_posterior = Gamma( 
                    alpha + math.prod(param_samples.size()),
                    beta + (param_samples ** 2).sum() / 2 
                )

                new_precisions[name] = precision_posterior.sample()

            print(new_precisions)

            model.ffnn[0].weight_precision.copy_(new_precisions["ffnn.0.weight"])
            model.ffnn[0].bias_precision.copy_(new_precisions["ffnn.0.bias"])
            model.ffnn[2].weight_precision.copy_(new_precisions["ffnn.2.weight"])
            model.ffnn[2].bias_precision.copy_(new_precisions["ffnn.2.bias"])

if __name__ == "__main__":

    main()  


#