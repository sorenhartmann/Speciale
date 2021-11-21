import torch
from src.models.base import ClassifierMixin, Model


class Densenet(Model):
    def __init__(self, load=True, num_classes=10, dropout=0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.net = torch.hub.load(
            "pytorch/vision:v0.10.0", "densenet121", pretrained=load, dropout=dropout
        )
        self.prepare_net()

    def prepare_net(self):

        in_features = self.net.classifier.in_features
        self.net.classifier = torch.nn.Linear(
            in_features=in_features, out_features=self.num_classes
        )

    def forward(self, x):
        return self.net.forward(x)

class DensenetClassifier(ClassifierMixin, Densenet):
    ...
