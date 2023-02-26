# pyright: reportMissingTypeStubs=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import ParameterGrid
from typing import NamedTuple, TypedDict

from protocols.protocols import get_protocol
from protocols.sample import ProtocolSampler
from semantic_content.pipeline import pipeline
from semantic_content.transformer import SemanticContentTransformer


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
    "num_colors": [10],
    "num_shapes": [10],
    "seed": [1],
    "sample_size": [1000],
})


class Parameters(TypedDict):
    protocol_name: str
    num_colors: int
    num_shapes: int
    seed: int
    sample_size: int


ParameterValues = NamedTuple("ParameterValues", [("protocol_name", str), (
    "num_colors", int), ("num_shapes", int), ("seed", int), ("sample_size", int),])


def get_parameter_values(parameters: Parameters) -> ParameterValues:
    return ParameterValues(parameters["protocol_name"], parameters["num_colors"], parameters["num_shapes"], parameters["seed"], parameters["sample_size"])


def get_result(parameters: Parameters) -> NDArray[np.int_]:
    protocol_name, num_colors, num_shapes, seed, sample_size = get_parameter_values(
        parameters)

    documents = ProtocolSampler(get_protocol(
        protocol_name, num_colors, num_shapes), seed).documents(sample_size)

    x, _y = SemanticContentTransformer(window_size=2).fit_transform(documents)

    return pipeline.fit_transform(x).flatten()
