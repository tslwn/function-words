# pyright: reportMissingTypeStubs=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import ParameterGrid
from typing import NamedTuple, TypedDict

from protocols import get_protocol, ProtocolName
from semantic_content.pipeline import make_pipeline, ScalerName
from semantic_content.transformer import SemanticContentTransformer


parameter_grid = ParameterGrid({
    "protocol_name": [
        "context",
        "diagonal",
        "english",
        "holistic",
        "negation",
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
    "scaler_name": [
        "standard",
    ]
})


class Parameters(TypedDict):
    protocol_name: ProtocolName
    num_colors: int
    num_shapes: int
    seed: int
    sample_size: int
    scaler_name: ScalerName


ParameterValues = NamedTuple("ParameterValues", [
    ("protocol_name", ProtocolName),
    ("num_colors", int),
    ("num_shapes", int),
    ("seed", int),
    ("sample_size", int),
    ("scaler_name", ScalerName)
])


def get_parameter_values(parameters: Parameters) -> ParameterValues:
    return ParameterValues(parameters["protocol_name"], parameters["num_colors"], parameters["num_shapes"], parameters["seed"], parameters["sample_size"], parameters["scaler_name"])


def get_result(parameters: Parameters) -> NDArray[np.int_]:
    protocol_name, num_colors, num_shapes, seed, sample_size, scaler_name = get_parameter_values(
        parameters)

    protocol = get_protocol(protocol_name, num_colors, num_shapes, seed=seed)

    documents = protocol.documents(sample_size)

    x, _y = SemanticContentTransformer(window_size=2).fit_transform(documents)

    pipeline = make_pipeline(scaler_name)

    return pipeline.fit_transform(x).flatten()
