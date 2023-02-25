# pyright: reportMissingTypeStubs=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
import numpy as np
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from train.abstract.corpus import AbstractCorpus
from train.bnc.bnc_corpus import BNCCorpus
from train.wiki.wiki_corpus import WikiCorpus
from train.semantic_content_transformer import SemanticContentTransformer
from typing import TypedDict

preprocessor = ColumnTransformer(
    remainder='passthrough',
    transformers=[('scaler', StandardScaler(), [0])],
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
    ]
)


class Parameters(TypedDict):
    corpus_name: str
    seed: int
    sample_size: float
    window_size: int


parameter_grid = ParameterGrid({
    "corpus_name": ["BNC", "Simple English Wikipedia"],
    "seed": [1, 2, 3, 4, 5],
    "sample_size": [0.001],
    "window_size": [11],
})


def get_parameters(parameters: Parameters) -> tuple[str, int, float, int]:
    corpus_name = parameters["corpus_name"]
    seed = parameters["seed"]
    sample_size = parameters["sample_size"]
    window_size = parameters["window_size"]

    return corpus_name, seed, sample_size, window_size


def get_parameters_str(parameters: Parameters) -> str:
    corpus_name, seed, sample_size, window_size = get_parameters(parameters)

    return f"{corpus_name}, seed={seed}, sample_size={sample_size}, window_size={window_size}"


def get_corpus(parameters: Parameters) -> AbstractCorpus:
    corpus_name, seed, sample_size, _window_size = get_parameters(parameters)

    if corpus_name == "BNC":
        return BNCCorpus(seed=seed, sample_size=sample_size)
    elif corpus_name == "Simple English Wikipedia":
        return WikiCorpus(seed=seed, sample_size=sample_size)
    else:
        raise NotImplementedError


def train(parameters: Parameters) -> tuple[NDArray[np.float_], NDArray[np.bool_]]:
    corpus = get_corpus(parameters)

    x, y = SemanticContentTransformer(
        window_size=parameters["window_size"]).fit_transform(corpus.documents())

    x = pipeline.fit_transform(x).flatten()

    return x, y
