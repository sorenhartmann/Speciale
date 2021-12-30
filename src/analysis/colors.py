import seaborn as sns
from enum import Enum
from src.utils import Run
from typing import Dict, List, Tuple
import pandas as pd

color_idx = {
    "dropout": 0,
    "map": 1,
    "vi_exp_weighted": 2,
    "vi": 3,
    "sghmc_var_est_interbatch": 4,
    "sghmc": 5,
    "sghmc_var_est": 6,
    "sghmc_var_est_adam": 7,
    "hmc_batched": 8,
    "hmc_full": 9,
}

RGB_TYPE = Tuple[float, float, float]
base_palette = sns.color_palette()


def get_color_idx(run: Run) -> int:

    inf_cfg = run.cfg.inference

    if "sampler" in inf_cfg:
        sampler_cfg = inf_cfg.sampler

        if not "SGHMC" in sampler_cfg._target_:
            if run.cfg.data.batch_size == run.cfg.data.dataset.n_samples:
                return color_idx["hmc_full"]
            else:
                return color_idx["hmc_batched"]

        if (
            "variance_estimator" not in sampler_cfg
            or "Constant" in sampler_cfg.variance_estimator._target_
        ):
            return color_idx["sghmc"]
        else:
            est_target = sampler_cfg.variance_estimator._target_

            if "InterBatch" in est_target:
                return color_idx["sghmc_var_est_interbatch"]
            elif "Adam" in est_target:
                return color_idx["sghmc_var_est_adam"]
            else:
                return color_idx["sghmc_var_est"]

    elif "Variational" in inf_cfg._target_:

        if not "kl_weighting_scheme" in inf_cfg:
            return color_idx["vi"]

        weight_scheme = inf_cfg.kl_weighting_scheme._target_

        if "Exponential" in weight_scheme:
            return color_idx["vi_exp_weighted"]
        else:
            return color_idx["vi"]

    else:
        if inf_cfg.get("use_map", False):
            return color_idx["map"]
        else:
            return color_idx["dropout"]


def get_color(run: Run) -> RGB_TYPE:
    return base_palette[get_color_idx(run)]

def _to_hex(color: RGB_TYPE) -> str:
    a = int(255*color[0])
    b = int(255*color[1])
    c = int(255*color[2])
    return f"{a:x}{b:x}{c:x}"




def get_colors(labeled_runs: Dict[str, Run]) -> Tuple[List[RGB_TYPE], List[str]]:

    palette = []
    hue_order = []

    labels_and_colors = pd.DataFrame(
        [
            {"i": get_color_idx(run), "color": get_color(run), "label": label}
            for label, run in labeled_runs.items()
        ]
    ).sort_values(["i", "label"])

    fixed = []
    for color, data in labels_and_colors.groupby("color"):
        if len(data) > 1:
            code = f"light:#{_to_hex(color)}"
            new_palette = sns.color_palette(code, len(data)+1)[1:]
            data.color =  new_palette
        fixed.append(data)
    
    labels_and_colors = pd.concat(fixed).sort_values(["i", "label"])

    palette = labels_and_colors.color.tolist()
    hue_order = labels_and_colors.label.tolist()

    return palette, hue_order
