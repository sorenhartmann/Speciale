import re
from src.utils import Run

def get_variance_estimator(run: Run) -> str:

    estimator = run.cfg["inference"]["sampler"]["variance_estimator"]["_target_"]
    match = re.match("src.inference.mcmc.variance_estimators.(.+)", estimator)
    assert match is not None
    estimator = match[1]
    return estimator