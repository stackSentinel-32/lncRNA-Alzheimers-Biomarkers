"""tests/test_stage15_models.py
Stage 15 — Tests for scripts/ml/models.py

Verifies:
  - All 5 model pipelines build without error.
  - Each pipeline produces valid probability outputs on synthetic data.
  - Preprocessing (imputation + scaling) is inside each pipeline.
  - SelectKBest k values are capped to n_features.
  - build_model_registry returns all expected model names.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ml.models import FEATURE_SETS, SEED, build_model_registry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_dataset():
    """Minimal synthetic dataset: 60 samples, 20 features, balanced classes."""
    X_arr, y_arr = make_classification(
        n_samples=60,
        n_features=20,
        n_informative=5,
        n_redundant=3,
        n_classes=2,
        random_state=SEED,
        weights=[0.5, 0.5],
    )
    sample_ids = [f"S{i:03d}" for i in range(60)]
    X = pd.DataFrame(X_arr, index=sample_ids, columns=[f"GENE{i}" for i in range(20)])
    y = pd.Series(y_arr, index=sample_ids, name="label")
    return X, y


@pytest.fixture
def dataset_with_missing(small_dataset):
    """Dataset with 10% missing values to test imputation."""
    X, y = small_dataset
    rng = np.random.default_rng(SEED)
    X_missing = X.copy().astype(float)
    mask = rng.random(X.shape) < 0.10
    X_missing[mask] = np.nan
    return X_missing, y


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildModelRegistry:
    def test_returns_all_model_names(self, small_dataset):
        X, _ = small_dataset
        registry = build_model_registry(n_features=X.shape[1])
        expected = {"logistic_baseline", "logistic_l1", "elastic_net",
                    "random_forest", "rbf_svm"}
        assert set(registry.keys()) == expected

    def test_raises_on_zero_features(self):
        with pytest.raises(ValueError, match="n_features"):
            build_model_registry(n_features=0)

    def test_pipeline_grid_tuples(self, small_dataset):
        X, _ = small_dataset
        registry = build_model_registry(n_features=X.shape[1])
        for name, (pipe, grid) in registry.items():
            assert hasattr(pipe, "fit"), f"{name} pipeline missing fit()"
            assert isinstance(grid, dict), f"{name} grid is not a dict"

    def test_first_steps_are_impute_and_scale(self, small_dataset):
        X, _ = small_dataset
        registry = build_model_registry(n_features=X.shape[1])
        for name, (pipe, _) in registry.items():
            step_names = [s[0] for s in pipe.steps]
            assert "impute" in step_names, f"{name}: 'impute' step missing"
            assert "scale" in step_names, f"{name}: 'scale' step missing"
            assert step_names.index("impute") < step_names.index("scale"), (
                f"{name}: 'impute' must precede 'scale'"
            )

    def test_select_k_capped_to_n_features(self):
        """k values in SelectKBest grids must not exceed n_features."""
        n_features = 7
        registry = build_model_registry(n_features=n_features)
        for name, (pipe, grid) in registry.items():
            if "select__k" in grid:
                for k_val in grid["select__k"]:
                    assert k_val <= n_features, (
                        f"{name}: select__k={k_val} exceeds n_features={n_features}"
                    )


class TestModelProbabilities:
    @pytest.mark.parametrize("model_name", [
        "logistic_baseline", "logistic_l1", "elastic_net",
        "random_forest", "rbf_svm",
    ])
    def test_pipeline_produces_valid_probabilities(self, model_name, small_dataset):
        X, y = small_dataset
        registry = build_model_registry(n_features=X.shape[1])
        pipe, _ = registry[model_name]
        # Use first grid value for each param
        from sklearn.model_selection import GridSearchCV, StratifiedKFold
        cv = GridSearchCV(
            pipe,
            {k: [v[0]] for k, v in registry[model_name][1].items()},
            cv=StratifiedKFold(2, shuffle=True, random_state=SEED),
            scoring="roc_auc",
            n_jobs=1,
        )
        cv.fit(X, y)
        proba = cv.predict_proba(X)[:, 1]
        assert proba.shape == (len(y),), f"{model_name}: wrong proba shape"
        assert np.all(proba >= 0) and np.all(proba <= 1), (
            f"{model_name}: probabilities outside [0,1]"
        )

    @pytest.mark.parametrize("model_name", [
        "logistic_baseline", "logistic_l1", "elastic_net",
        "random_forest", "rbf_svm",
    ])
    def test_handles_missing_values(self, model_name, dataset_with_missing):
        """Pipeline must handle NaN via imputation without raising."""
        X, y = dataset_with_missing
        registry = build_model_registry(n_features=X.shape[1])
        pipe, _ = registry[model_name]
        from sklearn.model_selection import GridSearchCV, StratifiedKFold
        cv = GridSearchCV(
            pipe,
            {k: [v[0]] for k, v in registry[model_name][1].items()},
            cv=StratifiedKFold(2, shuffle=True, random_state=SEED),
            scoring="roc_auc",
            n_jobs=1,
        )
        cv.fit(X, y)  # must not raise
        proba = cv.predict_proba(X)[:, 1]
        assert not np.isnan(proba).any(), f"{model_name}: NaN in probabilities"


class TestFeatureSetConstants:
    def test_feature_sets_tuple_contains_expected(self):
        assert "age_sex_only" in FEATURE_SETS
        assert "lncrna_only" in FEATURE_SETS
        assert "combined" in FEATURE_SETS
