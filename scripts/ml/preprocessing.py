
"""Leakage-safe preprocessing pipelines for Stage 14."""

import numpy as np

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_preprocessing_pipeline(feature_group="lncrna"):
    """
    Create an unfitted preprocessing pipeline.

    All imputers, encoders, variance filters and scalers are fitted
    only on the training data passed to fit().
    """
    valid_groups = {"demographic", "lncrna", "combined"}
    if feature_group not in valid_groups:
        raise ValueError(
            f"feature_group must be one of {sorted(valid_groups)}."
        )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    transformers = [
        (
            "numeric",
            numeric_pipeline,
            make_column_selector(dtype_include=np.number),
        )
    ]

    # Demographic and combined groups contain categorical Sex.
    if feature_group in {"demographic", "combined"}:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            )),
        ])

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                make_column_selector(dtype_exclude=np.number),
            )
        )

    column_transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline([
        ("impute_encode", column_transformer),
        ("low_variance", VarianceThreshold(threshold=0.0)),
        ("scale", StandardScaler()),
    ])
