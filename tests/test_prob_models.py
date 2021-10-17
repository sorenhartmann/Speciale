import pytest
import torch
from torch.nn import Linear

from src.inference.probabilistic import (ModuleWithPrior, ProbabilisticModel,
                                         attach_priors_,
                                         to_probabilistic_model_)


@pytest.fixture
def probabilistic_classifier(classifier):
    model = classifier
    to_probabilistic_model_(model)
    return model

def test_attach_priors():

    a = Linear(10, 10)
    b = Linear(10, 10)

    attach_priors_(a)

    assert type(a) is not Linear
    assert isinstance(a, Linear)
    assert isinstance(a, ModuleWithPrior)
    assert isinstance(a, ModuleWithPrior[Linear])
    assert hasattr(a, "priors")

    attach_priors_(b)

    assert type(a) is type(b)


def test_attach_priors_serialization(tmp_path):

    a = Linear(10, 10)

    attach_priors_(a)

    assert type(a) is not Linear
    assert isinstance(a, Linear)
    assert isinstance(a, ModuleWithPrior[Linear])
    assert hasattr(a, "priors")

    torch.save(a, tmp_path / "test.pkl")
    b: ModuleWithPrior[Linear] = torch.load(tmp_path / "test.pkl")
    assert type(b) is ModuleWithPrior[Linear]

    for (name_a, param_a), (name_b, param_b) in zip(
        a.named_parameters(), b.named_parameters()
    ):
        assert name_a == name_b
        assert torch.equal(param_a, param_b)

    for (name_a, buffer_a), (name_b, buffer_b) in zip(
        a.named_buffers(), b.named_buffers()
    ):
        assert name_a == name_b
        assert torch.equal(buffer_a, buffer_b)


def test_attach_prior_lob_prob(tmp_path):

    a = Linear(10, 10)
    attach_priors_(a)
    a: ModuleWithPrior[Linear]
    log_prob = a.prior_log_prob()
    assert torch.torch.is_tensor(log_prob)
    assert torch.isfinite(a.prior_log_prob())


def test_to_probabilistic_model(classifier):

    old_class = classifier.__class__
    to_probabilistic_model_(classifier)
    assert type(classifier) is ProbabilisticModel[old_class]


def test_batch_loglik(probabilistic_classifier, batch):

    x, y = batch
    log_lik = probabilistic_classifier.log_likelihood(x, y)

    assert torch.torch.is_tensor(log_lik)
    assert torch.isfinite(log_lik).all()


def test_prior(probabilistic_classifier):

    log_prior = probabilistic_classifier.log_prior()

    assert torch.torch.is_tensor(log_prior)
    assert torch.isfinite(log_prior)


def test_model_serialization(classifier, tmp_path):

    a = classifier

    old_class = a.__class__
    to_probabilistic_model_(a)
    torch.save(a.state_dict(), tmp_path / "test.pkl")

    b = old_class()
    to_probabilistic_model_(b)
    b.load_state_dict(torch.load(tmp_path / "test.pkl"))

    for (name_a, param_a), (name_b, param_b) in zip(
        a.named_parameters(), b.named_parameters()
    ):
        assert name_a == name_b
        assert torch.equal(param_a, param_b)

    for (name_a, buffer_a), (name_b, buffer_b) in zip(
        a.named_buffers(), b.named_buffers()
    ):
        assert name_a == name_b
        assert torch.equal(buffer_a, buffer_b)
