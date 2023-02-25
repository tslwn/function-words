# pyright: reportMissingTypeStubs=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
import math
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import seaborn
from train.train import Parameters, parameter_grid, train
from typing import cast

order = ["Content", "Function"]
palette = {"Content": "#4c72b0", "Function": "#dd8452"}


def subplots(len: int) -> tuple[int, int]:
    n = math.isqrt(len)

    if n * n == len:
        return n, n

    return math.ceil(len / n), n


def y_labels(values: NDArray[np.int_]) -> NDArray[np.str_]:
    return np.array(["Function" if value == 1 else "Content" for value in values])


ncols, nrows = subplots(len(parameter_grid))
fig, axes = plt.subplots(ncols=ncols, nrows=nrows, squeeze=False)

for col in range(ncols):
    for row in range(nrows):
        index = (col * nrows) + row

        if index >= len(parameter_grid):
            break

        parameters = cast(Parameters, parameter_grid[index])
        corpus_name = parameters["corpus_name"]
        sample_size = parameters["sample_size"]
        window_size = parameters["window_size"]

        ax = axes[row][col]
        ax.set_xlabel("Semantic content (normalised)")
        ax.set_ylabel("Word class")
        ax.set_title(
            f"{corpus_name}, sample size {sample_size}, window size {window_size}")

        x, y_int = train(parameters)
        y = y_labels(y_int)

        seaborn.stripplot(x=x, y=y, hue=y, ax=ax, order=order, palette=palette)

plt.show()
