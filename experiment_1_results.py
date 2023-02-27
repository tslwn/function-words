# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false

import numpy as np
from numpy.typing import NDArray
from typing import NamedTuple, TypedDict

from corpora import CorpusName, get_corpus
from semantic_content.pipeline import make_pipeline, ScalerName
from semantic_content.transformer import SemanticContentTransformer


class Parameters(TypedDict):
    corpus_name: CorpusName
    seed: int
    sample_size: float
    window_size: int
    scaler_name: ScalerName


ParameterValues = NamedTuple("ParameterValues", [
    ("corpus_name", CorpusName),
    ("seed", int),
    ("sample_size", float),
    ("window_size", int),
    ("scaler_name", ScalerName),
])


def get_parameter_values(parameters: Parameters) -> ParameterValues:
    return ParameterValues(parameters["corpus_name"], parameters["seed"], parameters["sample_size"], parameters["window_size"], parameters["scaler_name"])


Result = NamedTuple(
    "Result", [("x", NDArray[np.float_]), ("y", NDArray[np.bool_])])


def get_result(parameters: Parameters) -> Result:
    corpus_name, seed, sample_size, window_size, scaler_name = get_parameter_values(
        parameters)

    corpus = get_corpus(corpus_name, seed, sample_size)

    documents = corpus.documents()

    x, y = SemanticContentTransformer(
        window_size=window_size).fit_transform(documents)

    pipeline = make_pipeline(scaler_name)

    x = pipeline.fit_transform(x).flatten()

    return Result(x, y)
