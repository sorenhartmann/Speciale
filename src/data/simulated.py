import torch


class SimulatedDataset(torch.utils.data.TensorDataset):
    def __init__(
        self,
        coeffs: list[float],
        n_samples: int = 10,
        sigma: float = 1.0,
        train: bool = True,
        seed: int = 123,
    ):

        self.coeffs = torch.tensor(coeffs)
        self.sigma = sigma

        with torch.random.fork_rng():

            torch.manual_seed(seed)
            x = torch.linspace(-3, 3, n_samples)
            x += torch.randn_like(x) / 5

            X = torch.stack([x ** i for i in range(len(coeffs))], dim=-1)
            y = X @ torch.tensor(coeffs) + torch.randn_like(x) * sigma
            Y = y.unsqueeze(-1)

        super().__init__(X, Y)
