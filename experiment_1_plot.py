# pyright: reportMissingTypeStubs=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import seaborn
from sklearn.model_selection import ParameterGrid
from typing import cast

from experiment_1_results import get_result, get_parameter_values, Parameters, Result
from utils import plot

parameter_grid = ParameterGrid({
    "corpus_name": [
        "BNC",
        "Simple English Wikipedia"
    ],
    "seed": list(range(1)),
    "sample_size": [0.001],
    "window_size": [11],
    "scaler_name": [
        "standard",
    ]
})


ORDER = ["Content", "Function"]
PALETTE = {"Content": "#4c72b0", "Function": "#dd8452"}


def y_labels(values: NDArray[np.bool_]) -> NDArray[np.str_]:
    return np.array(["Function" if value == 1 else "Content" for value in values])


def callback(ax: plt.Axes, result: tuple[Parameters, Result]) -> None:
    parameters, (x, y_bool) = result

    y = y_labels(y_bool)

    corpus_name, _seed, sample_size, window_size, scaler_name = get_parameter_values(
        parameters)

    ax.set_title(
        f"{corpus_name}, {scaler_name}, N={sample_size}, W={window_size}")

    ax.set_xlabel("Semantic content")
    ax.set_ylabel("Word class")

    seaborn.stripplot(x=x, y=y, hue=y, ax=ax, order=ORDER, palette=PALETTE)


if __name__ == "__main__":
    results: list[tuple[Parameters, Result]] = []

    for parameters in parameter_grid:
        parameters = cast(Parameters, parameters)
        results.append((parameters, get_result(parameters)))

    plot(results, callback)
