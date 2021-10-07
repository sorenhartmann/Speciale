import pytest
from src.inference.mcmc import MCMCInference
from src.inference.sgd import SGDInference
from src.inference.vi import VariationalInference
from pytorch_lightning import Trainer
import torch
import copy


def dict_equal(dict_a, dict_b):

    shared_keys = set(dict_a) & set(dict_b)
    for key in shared_keys:
        a = dict_a[key]
        b = dict_b[key]
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            if not torch.equal(a, b):
                return False
        else:
            raise NotImplementedError

    return True


@pytest.mark.parametrize(
    "inference_cls", [SGDInference, VariationalInference, MCMCInference]
)
def test_fast_dev_run(datamodule, classifier, inference_cls):

    inference = inference_cls(classifier)
    a_state = copy.deepcopy(inference.state_dict())
    Trainer(fast_dev_run=True).fit(inference, datamodule)
    b_state = inference.state_dict()
    assert not dict_equal(a_state, b_state)
