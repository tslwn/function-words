# pyright: reportMissingTypeStubs=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
import math
import matplotlib.pyplot as plt
import seaborn
from sklearn.model_selection import ParameterGrid
from typing import cast, TypedDict
from artificial.protocols import get_protocol
from artificial.sample import ProtocolSampler
from train.semantic_content_transformer import SemanticContentTransformer
from train.train import pipeline


class Parameters(TypedDict):
    protocol_name: str
    num_colors: int
    num_shapes: int
    seed: int
    sample_size: int


parameter_grid = ParameterGrid({
    "protocol_name": [
        "english",
        "diagonal",
        "holistic",
        "ntc",
        "order",
        "random",
        "rotated",
        "tc",
    ],
    "num_colors": [1000],
    "num_shapes": [1000],
    "seed": [1],
    "sample_size": [100000],
})


def subplots(len: int) -> tuple[int, int]:
    n = math.isqrt(len)

    if n * n == len:
        return n, n

    return math.ceil(len / n), n


ncols, nrows = subplots(len(parameter_grid))
fig, axes = plt.subplots(ncols=ncols, nrows=nrows, squeeze=False)


for col in range(ncols):
    for row in range(nrows):
        index = (col * nrows) + row

        if index >= len(parameter_grid):
            break

        parameters = cast(Parameters, parameter_grid[index])

        protocol_name = parameters["protocol_name"]
        num_colors = parameters["num_colors"]
        num_shapes = parameters["num_shapes"]
        seed = parameters["seed"]
        sample_size = parameters["sample_size"]

        protocol = get_protocol(protocol_name, num_colors, num_shapes)
        sampler = ProtocolSampler(protocol, seed)
        documents = sampler.documents(sample_size)

        x, _y = SemanticContentTransformer(
            window_size=2).fit_transform(documents)

        x = pipeline.fit_transform(x).flatten()

        ax = axes[row][col]
        ax.set_xlabel("Semantic content (normalised)")
        ax.set_title(
            f"{protocol_name}, C={num_colors}, S={num_shapes}, N={sample_size}")

        seaborn.stripplot(x=x, ax=ax)

plt.show()
