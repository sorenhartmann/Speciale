import pytest
from hydra import compose, initialize
from hydra.utils import instantiate

from src.inference.base import InferenceModule

overrides = [
    [
    "model=mlp_classifier",
    "inference=sgd",
    "inference.lr=0.005",
    ],
    [
    "model=mlp_classifier",
    "inference=vi",
    "inference.lr=1e-3",
    "inference.n_samples=10",
    ],
    [
    "model=mlp_classifier",
    "inference=mcmc",
    "sampler=sghmc",
    "sample_container=fifo",
    "inference.burn_in=50",
    "sampler.lr=0.2e-5",
    "sample_container.max_items=10",
    "sample_container.keep_every=1",
    ],
    [
    "model=mlp_classifier",
    "inference=mcmc",
    ]
]


@pytest.mark.parametrize("overrides", overrides)
def test_flat_spec(overrides):
    overrides += ["data=mnist"]
    with initialize("../conf"):
        cfg = compose("inference_config", overrides=overrides)
        inference = instantiate(cfg.inference)
        assert isinstance(inference, InferenceModule)
