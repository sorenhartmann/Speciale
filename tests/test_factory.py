inference_config = ComponentConfig(
    name="mcmc",
    config={
        "model": ComponentConfig(name="mlp_classifer"),
        "sampler": ComponentConfig(name="sghmc", config={"lr": 50}),
        "sample_container": ComponentConfig(
            name="fifo", config={"max_items": 10, "keep_every": 1}
        ),
        "burn_in": 50,
    },
)

flat_spec = {
    "model": "mlp_classifer",
    "inference": "mcmc",
    "mcmc.sampler": "sghmc",
    "mcmc.sample_container": "fifo",
    "mcmc.burn_in": 50,
    "sghmc.lr": 0.2e-5,
    "fifo.max_items": 10,
    "fifo.keep_every": 1,
}


flat_spec = {
    "model": "mlp_classifer",
    "inference": "sgd",
    "sgd.lr": 0.004,

    # ... trainer args
}


a = {
    "mcmc.sampler": "sghmc",
    "mcmc.sampler.lr": 0.2e-5,
    "mcmc.sample_container": "fifo",
    "mcmc.sample_container.max_items": 10,
    "mcmc.sample_container.keep_every": 1,
    "mcmc.burn_in": 50,
}


a = {
    "mcmc.sampler": "sghmc",
    "mcmc.lr": 0.2e-5,
    "mcmc.sample_container": "fifo",
    "mcmc.sample_container.max_items": 10,
    "mcmc.sample_container.keep_every": 1,
    "mcmc.burn_in": 50,
}



flat_spec = {
    "model": "mlp_classifer",
    "inference": "mcmc",
    "mcmc.sampler": "sghmc",
    "mcmc.sampler.lr": 0.2e-5,
    "mcmc.sample_container": "fifo",
    "mcmc.sample_container.max_items": 10,
    "mcmc.sample_container.keep_every": 1,
    "mcmc.burn_in": 50,
}


a = {
    "mcmc.sampler": "sghmc",
    "mcmc.sampler.lr": 0.2e-5,
    "mcmc.sample_container": "fifo",
    "mcmc.sample_container.max_items": 10,
    "mcmc.sample_container.keep_every": 1,
    "mcmc.burn_in": 50,
}
a = {".".join(k.split(".")[1:]): v for k, v in a.items()}
