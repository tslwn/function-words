# pyright: reportUnknownMemberType=false

import math
import matplotlib.pyplot as plt
from typing import Callable, NamedTuple, TypeVar


Grid = NamedTuple("Grid", [("ncols", int), ("nrows", int)])


def get_subplot_grid(len: int) -> Grid:
    n = math.isqrt(len)
    if n * n == len:
        return Grid(n, n)
    else:
        return Grid(math.ceil(len / n), n)


T = TypeVar("T")


def plot(results: list[T], callback: Callable[[plt.Axes, T], None]) -> None:
    n = len(results)
    ncols, nrows = get_subplot_grid(n)

    _fig, axes = plt.subplots(ncols=ncols, nrows=nrows, squeeze=False)

    for col in range(ncols):
        for row in range(nrows):
            index = (col * nrows) + row
            if index >= n:
                break

            ax = axes[row][col]
            result = results[index]

            callback(ax, result)

    plt.show()
