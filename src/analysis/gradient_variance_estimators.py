from pathlib import Path
import torch
import pandas as pd

def get_variance_estimates(path: Path) -> pd.DataFrame:

    data = torch.load(path)
    observed_variance = list(x["observed_variance"] for x in data["estimates"].values())
    estimated_variance = list(
        x["estimated_variance"] for x in data["estimates"].values()
    )
    step_index = pd.Index(data["estimates"], name="step")
    parameter_index = pd.Index(data["log_idx"].numpy(), name="parameter")

    return pd.concat(
        {
            "observed_variance": pd.DataFrame(
                torch.stack(observed_variance).numpy(),
                index=step_index,
                columns=parameter_index,
            ),
            "estimated_variance": pd.DataFrame(
                torch.stack(estimated_variance).numpy(),
                index=step_index,
                columns=parameter_index,
            ),
        },
        names=["name"],
    ).sort_index()
