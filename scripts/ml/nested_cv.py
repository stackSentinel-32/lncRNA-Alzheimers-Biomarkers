"""scripts/ml/nested_cv.py
Stage 15 — Leakage-resistant nested cross-validation for the lncSignal-AD study.

Design:
  Outer loop  — 5-fold (or fewer if class size is small) stratified CV for
                unbiased performance estimation.
  Inner loop  — 3-fold stratified GridSearchCV for hyperparameter tuning.
  All imputation, scaling and feature selection are INSIDE the fitted pipeline.
  No test-fold data is ever seen by any preprocessing transformer.

Group-aware splitting:
  If `groups` is provided, StratifiedGroupKFold is used for the outer loop
  to ensure no patient contributes samples to both train and validation folds.

Reproducibility:
  Fixed random seeds derived from the project seed (42). The outer splitter
  uses SEED and inner uses SEED + 1 to avoid identical fold structure.

Exports:
  NestedCVResult  — dataclass holding OOF predictions and fold diagnostics
  run_nested_cv(X, y, models, groups=None) -> dict[str, NestedCVResult]
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedGroupKFold,
    StratifiedKFold,
)

SEED: int = 42
_MIN_CLASS_SIZE_FOR_5_FOLD: int = 20
_MIN_CLASS_SIZE_FOR_3_FOLD: int = 9


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class NestedCVResult:
    """Holds all output from nested CV for one model on one feature set.

    Attributes:
        model_name: Identifier string for the model.
        feature_set: Identifier string for the feature set.
        oof_probabilities: Out-of-fold predicted probabilities (index-aligned
            with the input X/y).
        fold_best_params: Per-outer-fold best hyperparameters chosen by inner CV.
        n_outer_folds: Actual number of outer folds used.
        n_inner_folds: Actual number of inner folds used.
        sample_ids: Index of X in the order used (for sanity checking).
    """

    model_name: str
    feature_set: str
    oof_probabilities: pd.Series  # index = sample_id, values in [0, 1]
    fold_best_params: list[dict[str, Any]] = field(default_factory=list)
    n_outer_folds: int = 5
    n_inner_folds: int = 3
    sample_ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Fold-count chooser
# ---------------------------------------------------------------------------

def _choose_fold_count(
    y: pd.Series,
    requested_outer: int = 5,
    requested_inner: int = 3,
) -> tuple[int, int]:
    """Choose outer and inner fold counts appropriate to class sizes.

    Args:
        y: Binary label series (0/1).
        requested_outer: Desired outer-fold count.
        requested_inner: Desired inner-fold count.

    Returns:
        (outer_folds, inner_folds) — may be reduced to prevent empty splits.
    """
    min_class = int(y.value_counts().min())

    if min_class < _MIN_CLASS_SIZE_FOR_3_FOLD:
        raise ValueError(
            f"Minimum class size is {min_class}; nested CV requires at least "
            f"{_MIN_CLASS_SIZE_FOR_3_FOLD} samples per class."
        )

    outer = min(requested_outer, min_class // 4)
    outer = max(outer, 2)  # at least 2-fold
    inner = min(requested_inner, (min_class - min_class // outer) // 2)
    inner = max(inner, 2)

    if outer < requested_outer:
        warnings.warn(
            f"Outer CV reduced from {requested_outer} to {outer} folds "
            f"due to small minimum class size ({min_class}).",
            stacklevel=3,
        )
    if inner < requested_inner:
        warnings.warn(
            f"Inner CV reduced from {requested_inner} to {inner} folds.",
            stacklevel=3,
        )

    return outer, inner


# ---------------------------------------------------------------------------
# Core nested CV runner
# ---------------------------------------------------------------------------

def run_nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    models: dict[str, tuple[Any, dict]],
    feature_set: str = "lncrna_only",
    groups: Optional[pd.Series] = None,
    outer_folds: int = 5,
    inner_folds: int = 3,
) -> dict[str, NestedCVResult]:
    """Run nested cross-validation for all models in the registry.

    Args:
        X: Feature matrix, shape (n_samples, n_features). Index = sample IDs.
        y: Binary labels (0=Control, 1=AD). Same index as X.
        models: dict from build_model_registry() — {name: (pipeline, grid)}.
        feature_set: Label for the feature set being evaluated.
        groups: Optional Series of group IDs (e.g., patient IDs) for
                group-aware outer splitting. Same index as X.
        outer_folds: Desired number of outer CV folds.
        inner_folds: Desired number of inner CV folds.

    Returns:
        dict mapping model_name -> NestedCVResult.

    Scientific note:
        OOF probabilities are the unbiased performance estimates. Final model
        selection should be based on these, not on inner-CV scores. The
        inner-CV is for hyperparameter selection only.

    Leakage safeguards:
        - X and y must share the same index; alignment is verified.
        - Preprocessing transformers are fitted ONLY on training indices.
        - No validation-fold data is passed to any .fit() call.
        - Groups ensure patient-level independence across folds.
    """
    # ----- Input validation -------------------------------------------------
    if not X.index.equals(y.index):
        raise ValueError("X and y must share the same index (sample IDs).")
    if y.nunique() != 2:
        raise ValueError("y must be binary (exactly two distinct values).")
    if groups is not None and not X.index.equals(groups.index):
        raise ValueError("groups must share the same index as X.")
    if X.shape[1] < 1:
        raise ValueError("X must have at least 1 feature column.")

    actual_outer, actual_inner = _choose_fold_count(y, outer_folds, inner_folds)

    # ----- Splitters --------------------------------------------------------
    if groups is not None:
        outer_splitter = StratifiedGroupKFold(
            n_splits=actual_outer, shuffle=True, random_state=SEED
        )
        split_kwargs: dict[str, Any] = {"groups": groups.values}
    else:
        outer_splitter = StratifiedKFold(
            n_splits=actual_outer, shuffle=True, random_state=SEED
        )
        split_kwargs = {}

    inner_splitter = StratifiedKFold(
        n_splits=actual_inner, shuffle=True, random_state=SEED + 1
    )

    results: dict[str, NestedCVResult] = {}

    for model_name, (pipeline, param_grid) in models.items():
        print(f"  [nested_cv] Running {model_name} / {feature_set} ...")

        oof = pd.Series(np.nan, index=X.index, dtype=float, name="oof_probability")
        fold_params: list[dict[str, Any]] = []

        splits = list(outer_splitter.split(X, y, **split_kwargs))

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_val = X.iloc[val_idx]

            # Inner CV for hyperparameter selection — fitted on training fold ONLY.
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring="roc_auc",
                cv=inner_splitter,
                n_jobs=1,
                refit=True,
                error_score=np.nan,
                verbose=0,
            )
            search.fit(X_train, y_train)

            if not hasattr(search, "best_estimator_"):
                raise RuntimeError(
                    f"GridSearchCV failed to fit for {model_name} fold {fold_idx}"
                )

            proba = search.predict_proba(X_val)[:, 1]
            oof.iloc[val_idx] = proba
            fold_params.append(
                {
                    "fold": fold_idx + 1,
                    "best_params": search.best_params_,
                    "best_inner_cv_auc": float(search.best_score_),
                }
            )

        # Verify all samples received an OOF prediction.
        n_missing = int(oof.isna().sum())
        if n_missing > 0:
            raise RuntimeError(
                f"{n_missing} samples lack OOF predictions for {model_name}. "
                "Check that all outer-fold indices are covered."
            )

        results[model_name] = NestedCVResult(
            model_name=model_name,
            feature_set=feature_set,
            oof_probabilities=oof,
            fold_best_params=fold_params,
            n_outer_folds=actual_outer,
            n_inner_folds=actual_inner,
            sample_ids=X.index.tolist(),
        )

    return results


# ---------------------------------------------------------------------------
# Final model fitter (after model family selection)
# ---------------------------------------------------------------------------

def fit_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    pipeline: Any,
    param_grid: dict,
    inner_folds: int = 5,
) -> Any:
    """Refit the selected model family on ALL discovery data.

    This is distinct from CV estimators and must be stored separately.
    It is used to generate a frozen model for future (sealed) external
    validation — it must NOT be evaluated on the same discovery data.

    Args:
        X: Full discovery feature matrix.
        y: Full discovery labels.
        pipeline: Pipeline object from build_model_registry().
        param_grid: Hyperparameter grid for the chosen model.
        inner_folds: Folds for final inner tuning.

    Returns:
        Fitted best_estimator_ from GridSearchCV.
    """
    inner_splitter = StratifiedKFold(
        n_splits=inner_folds, shuffle=True, random_state=SEED + 2
    )
    final_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=inner_splitter,
        n_jobs=1,
        refit=True,
        verbose=0,
    )
    final_search.fit(X, y)
    print(f"  [fit_final_model] Best params: {final_search.best_params_}")
    print(f"  [fit_final_model] Inner CV AUC: {final_search.best_score_:.4f}")
    return final_search.best_estimator_
