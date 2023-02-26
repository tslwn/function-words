
# pyright: reportMissingTypeStubs=false

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

preprocessor = ColumnTransformer(
    remainder='passthrough',
    transformers=[('scaler', StandardScaler(), [0])],
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
    ]
)
