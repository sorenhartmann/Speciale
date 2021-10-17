from src.inference.mcmc.var_estimators import WelfordEstimator
import torch



def test_welford_estimator():

    torch.manual_seed(123)
    data = torch.randn((100, 10))

    wf = WelfordEstimator()
    for x in data:
        wf.update(x)

    assert torch.allclose(wf.estimate(),  torch.var(data, 0))
   