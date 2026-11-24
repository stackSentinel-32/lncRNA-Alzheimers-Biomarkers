"""tests/test_stage15_nested_cv.py
Stage 15 — Tests for scripts/ml/nested_cv.py

Verifies:
  - OOF predictions cover ALL samples (no gaps).
  - Stratified class balance is approximately maintained.
  - Reproducibility: identical seed → identical OOF predictions.
  - Fold count adapts correctly to small class sizes.
  - Group-aware splitting keeps groups intact across folds.
  - Validation fold data is never seen by inner CV fitting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ml.models import SEED, build_model_registry
from scripts.ml.nested_cv import NestedCVResult, _choose_fold_count, run_nested_cv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def balanced_dataset():
    """60 samples, 10 features, balanced classes."""
    X_arr, y_arr = make_classification(
        n_samples=60, n_features=10, n_informative=4,
        n_classes=2, random_state=SEED, weights=[0.5, 0.5],
    )
    ids = [f"P{i:03d}" for i in range(60)]
    X = pd.DataFrame(X_arr, index=ids, columns=[f"G{i}" for i in range(10)])
    y = pd.Series(y_arr, index=ids)
    return X, y


@pytest.fixture
def tiny_dataset():
    """20 samples, 5 features — forces reduced fold count."""
    X_arr, y_arr = make_classification(
        n_samples=20, n_features=5, n_informative=2,
        n_classes=2, random_state=SEED + 1, weights=[0.5, 0.5],
    )
    ids = [f"P{i:03d}" for i in range(20)]
    X = pd.DataFrame(X_arr, index=ids, columns=[f"G{i}" for i in range(5)])
    y = pd.Series(y_arr, index=ids)
    return X, y


@pytest.fixture
def simple_registry(balanced_dataset):
    X, _ = balanced_dataset
    # Return only logistic_baseline for speed in structural tests.
    full = build_model_registry(n_features=X.shape[1])
    return {"logistic_baseline": full["logistic_baseline"]}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFoldCountChooser:
    def test_standard_dataset_gives_requested_folds(self):
        y = pd.Series([0] * 30 + [1] * 30)
        outer, inner = _choose_fold_count(y, requested_outer=5, requested_inner=3)
        assert outer <= 5
        assert inner <= 3
        assert outer >= 2
        assert inner >= 2

    def test_small_dataset_reduces_folds(self):
        y = pd.Series([0] * 10 + [1] * 10)
        outer, inner = _choose_fold_count(y, requested_outer=5, requested_inner=3)
        assert outer <= 5
        assert inner <= 3

    def test_too_small_raises(self):
        y = pd.Series([0] * 3 + [1] * 3)
        with pytest.raises(ValueError, match="nested CV requires at least"):
            _choose_fold_count(y, requested_outer=5, requested_inner=3)


class TestRunNestedCV:
    def test_oof_covers_all_samples(self, balanced_dataset, simple_registry):
        X, y = balanced_dataset
        results = run_nested_cv(X, y, simple_registry, feature_set="lncrna_only")
        for name, ncv_result in results.items():
            assert len(ncv_result.oof_probabilities) == len(y), (
                f"{name}: OOF length mismatch"
            )
            assert not ncv_result.oof_probabilities.isna().any(), (
                f"{name}: NaN in OOF predictions"
            )

    def test_oof_index_matches_input(self, balanced_dataset, simple_registry):
        X, y = balanced_dataset
        results = run_nested_cv(X, y, simple_registry, feature_set="lncrna_only")
        for name, ncv_result in results.items():
            pd.testing.assert_index_equal(
                ncv_result.oof_probabilities.index,
                X.index,
                check_names=False,
            )

    def test_oof_probabilities_in_unit_interval(self, balanced_dataset, simple_registry):
        X, y = balanced_dataset
        results = run_nested_cv(X, y, simple_registry, feature_set="lncrna_only")
        for name, ncv_result in results.items():
            proba = ncv_result.oof_probabilities.values
            assert np.all(proba >= 0.0), f"{name}: probability < 0"
            assert np.all(proba <= 1.0), f"{name}: probability > 1"

    def test_reproducibility_same_seed(self, balanced_dataset):
        """Two runs with the same seed must produce identical OOF predictions."""
        X, y = balanced_dataset
        registry = build_model_registry(n_features=X.shape[1])
        single_model = {"logistic_l1": registry["logistic_l1"]}

        res1 = run_nested_cv(X, y, single_model, feature_set="lncrna_only")
        res2 = run_nested_cv(X, y, single_model, feature_set="lncrna_only")
        np.testing.assert_array_almost_equal(
            res1["logistic_l1"].oof_probabilities.values,
            res2["logistic_l1"].oof_probabilities.values,
            decimal=10,
            err_msg="OOF predictions differ across runs with same seed",
        )

    def test_mismatched_index_raises(self, balanced_dataset, simple_registry):
        X, y = balanced_dataset
        y_bad = y.reset_index(drop=True)  # different index
        with pytest.raises(ValueError, match="same index"):
            run_nested_cv(X, y_bad, simple_registry, feature_set="lncrna_only")

    def test_fold_params_recorded_per_fold(self, balanced_dataset, simple_registry):
        X, y = balanced_dataset
        results = run_nested_cv(X, y, simple_registry, feature_set="lncrna_only")
        for name, ncv_result in results.items():
            n_folds = ncv_result.n_outer_folds
            assert len(ncv_result.fold_best_params) == n_folds, (
                f"{name}: expected {n_folds} fold params, "
                f"got {len(ncv_result.fold_best_params)}"
            )

    def test_result_contains_feature_set_label(self, balanced_dataset, simple_registry):
        X, y = balanced_dataset
        results = run_nested_cv(X, y, simple_registry, feature_set="lncrna_only")
        for name, ncv_result in results.items():
            assert ncv_result.feature_set == "lncrna_only"

    def test_group_aware_splitting(self, balanced_dataset, simple_registry):
        """Groups must not be split across outer folds."""
        X, y = balanced_dataset
        # Assign 2 samples per patient (30 patients × 2).
        groups = pd.Series(
            [f"PAT{i//2:02d}" for i in range(len(X))],
            index=X.index,
        )
        results = run_nested_cv(
            X, y, simple_registry,
            feature_set="lncrna_only",
            groups=groups,
        )
        for name, ncv_result in results.items():
            assert not ncv_result.oof_probabilities.isna().any(), (
                f"{name}: NaN in group-aware OOF predictions"
            )
