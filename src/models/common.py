
import math
import torch
from torch import nn

def flatten_tensors(tensors):
    return torch.cat([x.flatten() for x in tensors])

class BaysianModule(nn.Module):

    def log_prior(self):
        """Returns log p(theta)"""
        raise NotImplementedError

    def log_likelihood(self, x: torch.FloatTensor, y: torch.FloatTensor):
        """Returns log p(x | theta)"""
        raise NotImplementedError

    def theta(self):
        return flatten_tensors(self.parameters())

    def update_theta(self, theta):
        with torch.no_grad():
            a = 0
            for parameter in self.parameters():
                b = a + math.prod(parameter.shape)
                parameter.copy_(theta[a:b])
                a = b

