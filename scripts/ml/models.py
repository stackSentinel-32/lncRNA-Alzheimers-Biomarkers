"""scripts/ml/models.py
Stage 15 — Model registry for the lncSignal-AD nested cross-validation study.

Defines five scikit-learn pipelines and three feature-set configurations.
All preprocessing (imputation, scaling, supervised selection) is embedded
inside the pipeline so it is fitted only on training data within each fold.

Scientific note:
  Blood discovery (GSE63061) found only 3 FDR-significant probes, with one
  partial replication (MIR29B2CHG). AUC near 0.5 is a scientifically honest
  expected outcome. Models are kept modest in complexity.

Exports:
  build_model_registry(n_features) -> dict[str, tuple[Pipeline, dict]]
  FEATURE_SETS  — names of the three comparison feature sets
  SEED          — project-wide fixed random seed (42)
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Suppress sklearn 1.8+ deprecation warnings for penalty/probability
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ---------------------------------------------------------------------------
# Project constants
# ---------------------------------------------------------------------------

SEED: int = 42

# Three feature-set comparison labels used throughout Stage 15.
FEATURE_SETS: tuple[str, ...] = ("age_sex_only", "lncrna_only", "combined")


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------

def _common_steps() -> list[tuple[str, Any]]:
    """Imputation + scaling steps shared by every model.

    These are always the first steps in the pipeline so that feature scaling
    and missing-value handling are refitted on each training fold.
    """
    return [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]


def _selection_step(k: int) -> tuple[str, SelectKBest]:
    """Univariate F-test feature selection (inside fold; not a leakage source)."""
    return ("select", SelectKBest(score_func=f_classif, k=k))


def _logistic_baseline() -> tuple[Pipeline, dict[str, list]]:
    """Un-regularised logistic regression — demographic or simple baseline.

    No hyperparameter tuning needed; a single-entry grid is provided so
    the GridSearchCV interface works uniformly.
    """
    pipe = Pipeline(
        _common_steps()
        + [("model", LogisticRegression(
            penalty=None,
            solver="lbfgs",
            max_iter=4000,
            random_state=SEED,
        ))]
    )
    grid: dict[str, list] = {"model__max_iter": [4000]}  # trivial grid
    return pipe, grid


def _logistic_l1(n_features: int) -> tuple[Pipeline, dict[str, list]]:
    """L1-regularised logistic regression (LASSO).

    L1 penalty drives irrelevant probe coefficients to exactly zero, yielding
    an interpretable sparse panel. C is tuned in the inner CV.
    """
    pipe = Pipeline(
        _common_steps()
        + [("model", LogisticRegression(
            penalty="l1",
            solver="liblinear",
            max_iter=4000,
            random_state=SEED,
        ))]
    )
    grid: dict[str, list] = {
        "model__C": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
    }
    return pipe, grid


def _elastic_net(n_features: int) -> tuple[Pipeline, dict[str, list]]:
    """Elastic Net logistic regression (L1 + L2 mix).

    Appropriate when blood lncRNA probes are correlated: pure L1 picks one
    from each correlated group arbitrarily, while EN can retain several.
    Uses saga solver which supports both L1 and L2 simultaneously.
    """
    pipe = Pipeline(
        _common_steps()
        + [("model", LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            max_iter=6000,
            random_state=SEED,
        ))]
    )
    grid: dict[str, list] = {
        "model__C": [0.01, 0.1, 1.0, 10.0],
        "model__l1_ratio": [0.1, 0.5, 0.9],
    }
    return pipe, grid


def _random_forest(n_features: int) -> tuple[Pipeline, dict[str, list]]:
    """Random Forest with univariate pre-selection.

    SelectKBest inside the pipeline respects fold boundaries. Tree depth and
    minimum-leaf constraints keep the model modest for the small sample size.
    """
    k_max = min(20, n_features)
    k_mid = min(10, n_features)
    k_small = min(5, n_features)
    k_grid = sorted(set([k_small, k_mid, k_max]))

    pipe = Pipeline(
        _common_steps()
        + [_selection_step(k_small)]
        + [("model", RandomForestClassifier(
            random_state=SEED,
            n_jobs=1,
            class_weight="balanced",
        ))]
    )
    grid: dict[str, list] = {
        "select__k": k_grid,
        "model__n_estimators": [200, 400],
        "model__max_depth": [3, 5],
        "model__min_samples_leaf": [3, 7],
    }
    return pipe, grid


def _rbf_svm(n_features: int) -> tuple[Pipeline, dict[str, list]]:
    """RBF-kernel Support Vector Machine with univariate pre-selection.

    SVM is sensitive to irrelevant features and scale; both are addressed by
    the pipeline. class_weight='balanced' accounts for class imbalance.
    """
    k_max = min(20, n_features)
    k_mid = min(10, n_features)
    k_small = min(5, n_features)
    k_grid = sorted(set([k_small, k_mid, k_max]))

    pipe = Pipeline(
        _common_steps()
        + [_selection_step(k_small)]
        + [("model", SVC(
            kernel="rbf",
            probability=True,
            random_state=SEED,
            class_weight="balanced",
        ))]
    )
    grid: dict[str, list] = {
        "select__k": k_grid,
        "model__C": [0.1, 1.0, 10.0],
        "model__gamma": ["scale", 0.01, 0.001],
    }
    return pipe, grid


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def build_model_registry(n_features: int) -> dict[str, tuple[Pipeline, dict]]:
    """Return the full model registry keyed by model name.

    Args:
        n_features: Number of input features (used to cap SelectKBest k values).

    Returns:
        dict mapping model name -> (pipeline, hyperparameter_grid).
        All models include imputation and scaling as the first two steps.

    Usage::
        from scripts.ml.models import build_model_registry
        registry = build_model_registry(n_features=X.shape[1])
        for name, (pipe, grid) in registry.items():
            ...
    """
    if n_features < 1:
        raise ValueError(f"n_features must be >= 1, got {n_features}")

    return {
        "logistic_baseline": _logistic_baseline(),
        "logistic_l1": _logistic_l1(n_features),
        "elastic_net": _elastic_net(n_features),
        "random_forest": _random_forest(n_features),
        "rbf_svm": _rbf_svm(n_features),
    }
