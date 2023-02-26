# pyright: reportMissingTypeStubs=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import ParameterGrid
from typing import NamedTuple, TypedDict

from corpora import get_corpus
from semantic_content.pipeline import pipeline
from semantic_content.transformer import SemanticContentTransformer


parameter_grid = ParameterGrid({
    "corpus_name": ["BNC", "Simple English Wikipedia"],
    "seed": [1],
    "sample_size": [0.001],
    "window_size": [11],
})


class Parameters(TypedDict):
    corpus_name: str
    seed: int
    sample_size: float
    window_size: int


ParameterValues = NamedTuple("ParameterValues", [
    ("corpus_name", str),
    ("seed", int),
    ("sample_size", float),
    ("window_size", int),
])


def get_parameter_values(parameters: Parameters) -> ParameterValues:
    return ParameterValues(parameters["corpus_name"], parameters["seed"], parameters["sample_size"], parameters["window_size"])


Result = NamedTuple(
    "Result", [("x", NDArray[np.float_]), ("y", NDArray[np.bool_])])


def get_result(parameters: Parameters) -> Result:
    corpus_name, seed, sample_size, window_size = get_parameter_values(
        parameters)

    corpus = get_corpus(corpus_name, seed, sample_size)

    documents = corpus.documents()

    x, y = SemanticContentTransformer(
        window_size=window_size).fit_transform(documents)

    x = pipeline.fit_transform(x).flatten()

    return Result(x, y)
