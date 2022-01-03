import re
from typing import Any
from src.utils import Run
import pandas as pd 

def get_variance_estimator(run: Run) -> str:

    estimator = run.cfg["inference"]["sampler"]["variance_estimator"]["_target_"]
    match = re.match("src.inference.mcmc.variance_estimators.(.+)", estimator)
    assert match is not None
    estimator = match[1]
    return estimator

def add_column_level_(dataframe: pd.DataFrame, name: str) -> pd.DataFrame:
    dataframe.columns = pd.MultiIndex.from_product([[name], dataframe.columns])
    return dataframe

def format_as_percent(rel_err: float) -> str:
    return f"{100*rel_err:2.2f}\%"

def embolden_(dataframe: pd.DataFrame, index: Any) -> pd.DataFrame:

    dataframe.loc[index] = dataframe.loc[index].map(lambda x: f"\\textbf{{{x}}}")
    return dataframe.rename(index=lambda x: x if x != index else f"\\textbf{{{x}}}")

def format_rate_with_95_ci(
    data: pd.DataFrame,
    rate_col: str,
    count_col: str,
) -> pd.DataFrame:

    from math import sqrt

    def format_ci(rate: float, count: int) -> str:
        pm = sqrt(rate * (1 - rate) / count) * 1.96
        return f"{100*rate:.2f} $\\pm$ {100*pm:.2f}~\\%"

    return data.apply(lambda x: format_ci(x[rate_col], x[count_col]), axis=1)


