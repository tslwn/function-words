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
from semantic_content.pipeline import make_pipeline
from semantic_content.transformer import SemanticContentTransformer


parameter_grid = ParameterGrid({
    "protocol_name": [
        "english",
        # "diagonal",
        # "holistic",
        "ntc",
        # "order",
        # "random",
        # "rotated",
        "tc",
    ],
    "num_colors": [100],
    "num_shapes": [100],
    "seed": [1],
    "sample_size": [100000],
    "scaler_name": [
        "none",
        "standard",
        "min_max",
        "max_abs",
        "robust",
        "yeo_johnson",
        # "box_cox",
        "quantile_uniform",
        "quantile_normal",
        # "normal",
    ]
})


class Parameters(TypedDict):
    protocol_name: str
    num_colors: int
    num_shapes: int
    seed: int
    sample_size: int
    scaler_name: str


ParameterValues = NamedTuple("ParameterValues", [("protocol_name", str), (
    "num_colors", int), ("num_shapes", int), ("seed", int), ("sample_size", int), ("scaler_name", str)])


def get_parameter_values(parameters: Parameters) -> ParameterValues:
    return ParameterValues(parameters["protocol_name"], parameters["num_colors"], parameters["num_shapes"], parameters["seed"], parameters["sample_size"], parameters["scaler_name"])


def get_result(parameters: Parameters) -> NDArray[np.int_]:
    protocol_name, num_colors, num_shapes, seed, sample_size, scaler_name = get_parameter_values(
        parameters)

    protocol = get_protocol(protocol_name, num_colors, num_shapes)

    documents = ProtocolSampler(protocol, seed).documents(sample_size)

    x, _y = SemanticContentTransformer(window_size=2).fit_transform(documents)

    pipeline = make_pipeline(scaler_name)

    return pipeline.fit_transform(x).flatten()
