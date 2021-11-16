
import torch
from torch.nn import Linear, Sequential, ReLU
from src.bayesian.core import *
from src.inference.vi import VariationalInference
from src.inference.mcmc import MCMCInference
from src.models.base import Model, ClassifierMixin
from pytorch_lightning import Trainer
from src.inference.mcmc.samplers import SGHMC

class Net(ClassifierMixin, Model, Sequential):
    pass
        
net = Net(Linear(4, 10), ReLU(), Linear(10, 3))

from sklearn.datasets import load_iris
iris = load_iris()
X = torch.tensor(iris["data"], dtype=torch.float32)
y = torch.tensor(iris["target"])
dataset = torch.utils.data.TensorDataset(X, y)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [100, 50])


inference = MCMCInference(
    net,
    sampler=SGHMC(lr=1e-4, alpha=0.01),
    steps_per_sample=10,
    burn_in=10
    )


trainer = Trainer(max_epochs=500)
trainer.fit(
    inference,
    train_dataloaders=torch.utils.data.DataLoader(train_dataset, batch_size=50, shuffle=True),
    val_dataloaders=torch.utils.data.DataLoader(test_dataset, batch_size=25),
)

