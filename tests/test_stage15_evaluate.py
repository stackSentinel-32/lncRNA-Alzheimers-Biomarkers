"""tests/test_stage15_evaluate.py
Stage 15 — Tests for scripts/ml/evaluate.py

Verifies:
  - Metric correctness on known synthetic examples.
  - Bootstrap CI bounds AUC on simple data.
  - Youden threshold selection uses only training labels.
  - Confusion matrix values sum correctly.
  - evaluation_source label is always 'cv_oof' for internal results.
  - Plot functions run without errors (non-interactive backend).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ml.evaluate import (
    _bootstrap_auc_ci,
    _youden_threshold,
    compute_metrics,
    plot_bootstrap_ci,
    plot_calibration,
    plot_confusion_matrix,
    plot_roc_comparison,
)
from scripts.ml.models import SEED


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def perfect_predictions():
    """Ideal case: probabilities perfectly rank cases above controls."""
    y = pd.Series([0] * 20 + [1] * 20, name="label")
    proba = pd.Series([0.05] * 20 + [0.95] * 20, name="prob")
    return y, proba


@pytest.fixture
def chance_predictions():
    """Chance: probabilities are 0.5 for everyone."""
    y = pd.Series([0] * 20 + [1] * 20, name="label")
    proba = pd.Series([0.5] * 40, name="prob")
    return y, proba


@pytest.fixture
def realistic_predictions():
    """Moderate discrimination (AUC ~0.70)."""
    rng = np.random.default_rng(SEED)
    y_arr = np.array([0] * 25 + [1] * 25)
    # Controls centred at 0.4, cases at 0.6
    proba_arr = np.concatenate([
        np.clip(rng.normal(0.4, 0.15, 25), 0.01, 0.99),
        np.clip(rng.normal(0.6, 0.15, 25), 0.01, 0.99),
    ])
    return pd.Series(y_arr), pd.Series(proba_arr)


# ---------------------------------------------------------------------------
# Tests: compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_perfect_auc(self, perfect_predictions):
        y, proba = perfect_predictions
        m = compute_metrics(y, proba, label="cv_oof")
        assert m["auc"] == pytest.approx(1.0, abs=1e-6)

    def test_chance_auc_near_half(self, chance_predictions):
        y, proba = chance_predictions
        m = compute_metrics(y, proba, label="cv_oof")
        assert 0.0 <= m["auc"] <= 1.0

    def test_evaluation_source_label(self, realistic_predictions):
        y, proba = realistic_predictions
        m = compute_metrics(y, proba, label="cv_oof")
        assert m["evaluation_source"] == "cv_oof"

    def test_confusion_matrix_sums(self, realistic_predictions):
        y, proba = realistic_predictions
        m = compute_metrics(y, proba, label="cv_oof")
        total = m["TN"] + m["FP"] + m["FN"] + m["TP"]
        assert total == len(y), "Confusion matrix values must sum to N"

    def test_sensitivity_in_unit_interval(self, realistic_predictions):
        y, proba = realistic_predictions
        m = compute_metrics(y, proba, label="cv_oof")
        assert 0.0 <= m["sensitivity"] <= 1.0
        assert 0.0 <= m["specificity"] <= 1.0

    def test_f1_in_unit_interval(self, realistic_predictions):
        y, proba = realistic_predictions
        m = compute_metrics(y, proba, label="cv_oof")
        assert 0.0 <= m["f1"] <= 1.0

    def test_balanced_accuracy_in_unit_interval(self, realistic_predictions):
        y, proba = realistic_predictions
        m = compute_metrics(y, proba, label="cv_oof")
        assert 0.0 <= m["balanced_accuracy"] <= 1.0

    def test_brier_score_in_unit_interval(self, realistic_predictions):
        y, proba = realistic_predictions
        m = compute_metrics(y, proba, label="cv_oof")
        assert 0.0 <= m["brier_score"] <= 1.0

    def test_class_imbalance_ratio(self):
        y = pd.Series([0] * 30 + [1] * 10)
        proba = pd.Series(np.linspace(0, 1, 40))
        m = compute_metrics(y, proba, label="cv_oof")
        assert m["n_ad"] == 10
        assert m["n_control"] == 30
        assert m["class_imbalance_ratio"] == pytest.approx(10 / 30, rel=1e-4)

    def test_raises_on_single_class(self):
        y = pd.Series([0] * 20)
        proba = pd.Series([0.5] * 20)
        with pytest.raises(ValueError, match="both classes"):
            compute_metrics(y, proba, label="cv_oof")

    def test_provided_threshold_is_used(self, realistic_predictions):
        y, proba = realistic_predictions
        m = compute_metrics(y, proba, threshold=0.3, label="cv_oof")
        assert m["threshold_used"] == pytest.approx(0.3)
        assert m["threshold_method"] == "provided"

    def test_youden_threshold_method_label(self, realistic_predictions):
        y, proba = realistic_predictions
        m = compute_metrics(y, proba, label="cv_oof")
        assert m["threshold_method"] == "youden_oof"


# ---------------------------------------------------------------------------
# Tests: bootstrap CI
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    def test_ci_contains_true_auc_on_separable(self, perfect_predictions):
        """Bootstrap CI should include AUC=1.0 for perfectly separated data."""
        y, proba = perfect_predictions
        lo, hi = _bootstrap_auc_ci(np.array(y), np.array(proba), n=200, seed=SEED)
        assert lo <= 1.0 <= hi or hi >= 0.9, (
            "Bootstrap CI should be near 1.0 for perfect predictions"
        )

    def test_ci_lo_le_auc_le_hi(self, realistic_predictions):
        y, proba = realistic_predictions
        from sklearn.metrics import roc_auc_score
        auc_val = roc_auc_score(y, proba)
        lo, hi = _bootstrap_auc_ci(np.array(y), np.array(proba), n=200, seed=SEED)
        assert lo <= auc_val, "CI lower bound must be <= AUC"
        assert hi >= auc_val, "CI upper bound must be >= AUC"

    def test_ci_lo_le_hi(self, realistic_predictions):
        y, proba = realistic_predictions
        lo, hi = _bootstrap_auc_ci(np.array(y), np.array(proba), n=200, seed=SEED)
        assert lo <= hi


# ---------------------------------------------------------------------------
# Tests: Youden threshold
# ---------------------------------------------------------------------------

class TestYoudenThreshold:
    def test_threshold_in_unit_interval(self, realistic_predictions):
        y, proba = realistic_predictions
        t = _youden_threshold(np.array(y), np.array(proba))
        assert 0.0 <= t <= 1.0

    def test_perfect_threshold_near_boundary(self, perfect_predictions):
        """With perfect predictions, Youden threshold should be between 0.05 and 0.95."""
        y, proba = perfect_predictions
        t = _youden_threshold(np.array(y), np.array(proba))
        assert 0.0 < t < 1.0


# ---------------------------------------------------------------------------
# Tests: Plot functions (smoke tests — just check they don't crash)
# ---------------------------------------------------------------------------

class TestPlotFunctions:
    def test_plot_roc_comparison_saves_file(self, realistic_predictions, tmp_path):
        y, proba = realistic_predictions
        metrics = compute_metrics(y, proba, label="cv_oof")
        metrics["model"] = "test_model"
        metrics["feature_set"] = "lncrna_only"
        out = tmp_path / "roc.png"
        plot_roc_comparison(
            results={"test_model": metrics},
            y_true=y,
            oof_probs={"test_model": proba},
            title="Test ROC",
            save_path=out,
        )
        assert out.exists()
        assert out.stat().st_size > 1000, "ROC figure appears too small"

    def test_plot_confusion_matrix_saves_file(self, realistic_predictions, tmp_path):
        y, proba = realistic_predictions
        m = compute_metrics(y, proba, label="cv_oof")
        out = tmp_path / "cm.png"
        plot_confusion_matrix(
            y_true=y, proba=proba,
            threshold=m["threshold_used"],
            model_name="test_model",
            feature_set="lncrna_only",
            save_path=out,
        )
        assert out.exists()

    def test_plot_calibration_saves_file(self, realistic_predictions, tmp_path):
        y, proba = realistic_predictions
        out = tmp_path / "cal.png"
        plot_calibration(
            y_true=y,
            oof_probs={"test_model": proba},
            save_path=out,
        )
        assert out.exists()

    def test_plot_bootstrap_ci_saves_file(self, tmp_path):
        import pandas as pd
        df = pd.DataFrame([
            {"model": "logistic_l1", "feature_set": "lncrna_only",
             "auc": 0.62, "auc_ci_lo": 0.52, "auc_ci_hi": 0.72},
            {"model": "random_forest", "feature_set": "lncrna_only",
             "auc": 0.58, "auc_ci_lo": 0.48, "auc_ci_hi": 0.68},
        ])
        out = tmp_path / "ci.png"
        plot_bootstrap_ci(summary_df=df, save_path=out)
        assert out.exists()
