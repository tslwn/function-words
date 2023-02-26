
# pyright: reportMissingTypeStubs=false

from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer, PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler
from typing import Any


class IdentityScaler(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, _X: Any):
        return self

    def transform(self, X: Any) -> Any:
        return X


scalers = {
    "none": IdentityScaler(),
    "standard": StandardScaler(),
    "min_max": MinMaxScaler(),
    "max_abs": MaxAbsScaler(),
    "robust": RobustScaler(quantile_range=(25, 75)),
    "yeo_johnson": PowerTransformer(method="yeo-johnson"),
    "box_cox": PowerTransformer(method="box-cox"),
    "quantile_uniform": QuantileTransformer(output_distribution="uniform"),
    "quantile_normal": QuantileTransformer(output_distribution="normal"),
    "normal": Normalizer(),
}


def make_pipeline(scaler_name: str) -> Pipeline:
    scaler = scalers.get(scaler_name)
    if scaler is None:
        raise NotImplementedError

    preprocessor = ColumnTransformer(
        remainder="passthrough",
        transformers=[("scaler", scaler, [0])],
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
        ]
    )

    return pipeline
