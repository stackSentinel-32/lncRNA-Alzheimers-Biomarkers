"""Training-fold-only LASSO feature selection and stability analysis."""

import numpy as np
import pandas as pd
import sklearn
from packaging.version import Version

from sklearn.base import clone
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from scripts.ml.preprocessing import make_preprocessing_pipeline


class _LassoSelection(SelectFromModel):
    """Keep empty support inspectable, but stop before fitting a downstream model."""

    def transform(self, X):
        if not self.get_support().any():
            raise ValueError("LASSO selected zero features on this training fold.")
        return super().transform(X)


def make_lasso_selector(feature_group="lncrna", C=0.1):
    """
    Return an unfitted preprocessing + L1 logistic regression selector.

    C controls regularization strength. Smaller C means stronger
    regularization. No feature selection is performed before fit().
    """
    if C <= 0:
        raise ValueError("C must be positive.")

    # sklearn 1.8 moved the L1/L2 choice from penalty to l1_ratio.
    l1_params = {"l1_ratio": 1.0} if Version(sklearn.__version__) >= Version("1.8") else {"penalty": "l1"}
    lasso = LogisticRegression(
        **l1_params,
        solver="liblinear",
        C=C,
        class_weight="balanced",
        max_iter=5000,
        random_state=42,
    )

    return Pipeline([
        ("preprocessing", make_preprocessing_pipeline(feature_group)),
        ("selection", _LassoSelection(
            estimator=lasso,
            threshold=1e-5,  # sklearn's L1 coefficient-magnitude tolerance
        )),
    ])


def selected_feature_names(fitted_pipeline):
    """Return feature names retained by a fitted LASSO pipeline."""
    preprocessing = fitted_pipeline.named_steps["preprocessing"]

    names = preprocessing.named_steps[
        "impute_encode"
    ].get_feature_names_out()

    variance_mask = preprocessing.named_steps[
        "low_variance"
    ].get_support()

    names = np.asarray(names)[variance_mask]

    selection_mask = fitted_pipeline.named_steps[
        "selection"
    ].get_support()

    return names[selection_mask].tolist()


def selection_stability(
    X,
    y,
    pipeline,
    cv,
    groups=None,
):
    """
    Measure how frequently each feature is selected across training folds.

    The selector is cloned and fitted independently on each training fold.
    Pass groups when the CV splitter requires subject/cohort grouping.
    """
    counts = {}
    fold_results = []

    splits = cv.split(X, y, groups) if groups is not None else cv.split(X, y)

    for fold_number, (train_idx, test_idx) in enumerate(splits, start=1):
        # Only the training partition is used to fit preprocessing/selection.
        fitted = clone(pipeline)
        fitted.fit(X.iloc[train_idx], y.iloc[train_idx])

        features = selected_feature_names(fitted)

        for feature in features:
            counts[feature] = counts.get(feature, 0) + 1

        fold_results.append({
            "fold": fold_number,
            "training_samples": len(train_idx),
            "heldout_samples": len(test_idx),
            "selected_feature_count": len(features),
        })

    total_folds = len(fold_results)
    if total_folds == 0:
        raise ValueError("The CV splitter produced no folds.")

    stability = pd.DataFrame([
        {
            "feature_id": feature,
            "selected_folds": count,
            "selection_frequency": count / total_folds,
        }
        for feature, count in counts.items()
    ], columns=["feature_id", "selected_folds", "selection_frequency"])

    if not stability.empty:
        stability = stability.sort_values(
            ["selection_frequency", "selected_folds", "feature_id"],
            ascending=[False, False, True],
        ).reset_index(drop=True)

    return stability, pd.DataFrame(fold_results)
