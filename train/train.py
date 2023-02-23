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
    sample_size: float
    window_size: int


parameter_grid = ParameterGrid({
    "corpus_name": ["Simple English Wikipedia"],
    "sample_size": [0.001],
    "window_size": [3],
})


def get_corpus(corpus_name: str, sample_size: float) -> AbstractCorpus:
    if corpus_name == "BNC":
        return BNCCorpus(sample_size=sample_size)
    elif corpus_name == "Simple English Wikipedia":
        return WikiCorpus(sample_size=sample_size)
    else:
        raise NotImplementedError


def train(parameters: Parameters) -> tuple[NDArray[np.float_], NDArray[np.int_]]:
    corpus_name = parameters["corpus_name"]
    sample_size = parameters["sample_size"]
    window_size = parameters["window_size"]

    corpus = get_corpus(corpus_name, sample_size)

    x, y = SemanticContentTransformer(
        window_size=window_size).fit_transform(corpus.documents())

    x = pipeline.fit_transform(x).flatten()

    return x, y
