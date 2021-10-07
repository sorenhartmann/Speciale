import pytest
from src.inference.base import InferenceModule
from src.inference.factory import inference_from_config

flat_specs = [
    {
    "model": "mlp_classifier",
    "inference": "sgd",
    "sgd.lr": 0.005,
    },
    {
    "model": "mlp_classifier",
    "inference": "vi",
    "vi.lr": 1e-3,
    "vi.n_samples": 10,
    "vi.prior": None,  # Not implemented
    },
    {
    "model": "mlp_classifier",
    "inference": "mcmc",
    "mcmc.sampler": "sghmc",
    "mcmc.sample_container": "fifo",
    "mcmc.burn_in": 50,
    "sghmc.lr": 0.2e-5,
    "fifo.max_items": 10,
    "fifo.keep_every": 1,
    },
    {
    "model": "mlp_classifier",
    "inference": "mcmc"
    }
]

@pytest.mark.parametrize("flat_spec", flat_specs)
def test_flat_spec(flat_spec):
    inf = inference_from_config(flat_spec)
    assert isinstance(inf, InferenceModule)
