# pyright: reportMissingTypeStubs=false
# pyright: reportUnknownVariableType=false

import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.model_selection import ParameterGrid
from typing import cast

from experiment_1_results import get_parameter_values, get_result, Parameters

parameter_grid = ParameterGrid({
    "corpus_name": [
        "BNC",
        "Simple English Wikipedia"
    ],
    "seed": list(range(10)),
    "sample_size": [0.001],
    "window_size": [11],
    "scaler_name": [
        "standard",
    ]
})

if __name__ == "__main__":
    results = []

    for parameters in parameter_grid:
        parameters = cast(Parameters, parameters)

        x, y = get_result(parameters)

        x_f = [value for (value, f) in zip(x, y) if f]
        skew_f = skew(x_f)
        kurtosis_f = kurtosis(x_f)

        x_c = [value for (value, f) in zip(x, y) if not f]
        skew_c = skew(x_c)
        kurtosis_c = kurtosis(x_c)

        corpus_name, seed, sample_size, window_size, scaler_name = get_parameter_values(
            parameters)

        results.append((corpus_name, seed, sample_size, window_size,
                       scaler_name, skew_f, kurtosis_f, skew_c, kurtosis_c))

    df = pd.DataFrame(results, columns=["Corpus", "Random seed", "Sample size",
                      "Collocation window size", "Scaling", "Skewness (function words)", "Kurtosis (function words)", "Skewness (content words)", "Kurtosis (content words)"])

    df.to_csv("results/experiment_1_skew.csv")
