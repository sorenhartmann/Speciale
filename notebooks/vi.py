
from src.models.mlp import MLPClassifier
from src.inference.probabilistic import to_probabilistic_model_
from src.utils import ParameterView
from torch.distributions.normal import Normal
from src.data.mnist import MNISTDataModule

import torch





model = MLPClassifier()
to_probabilistic_model_(model)
view = ParameterView(model)
view.apply(lambda x: x.register_hook(print))
view.apply(lambda x: x.requires_grad_(False))

inference = torch.nn.Module()
inference.register_parameter("rho", torch.nn.Parameter(torch.zeros(view.n_params),))
inference.register_parameter("mu", torch.nn.Parameter(torch.zeros(view.n_params)))
inference.model = model

optimizer = torch.optim.SGD((inference.rho, inference.mu), lr=0.01)

dm = MNISTDataModule()
dm.setup()

for i, (x, y) in enumerate(dm.train_dataloader()):

    sigma = torch.log(1 + inference.rho.exp())
    eps = torch.randn_like(inference.mu)
    w = inference.mu + sigma * eps
    view[:] = w
    elbo = (
        Normal(inference.mu, sigma).log_prob(w).sum()
        - model.log_likelihood(x, y).sum()
        - model.log_prior()
    )

    view.apply(lambda x: x.retain_grad())
    elbo.backward()

    with torch.no_grad():
        inference.mu.grad.add_(view.flat_grad)
        inference.rho.grad.add_(view.flat_grad * (eps / (1 + torch.exp(-inference.rho))))
    
    inference.model.zero_grad(set_to_none=True)

    optimizer.step()
    optimizer.zero_grad()


