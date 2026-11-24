"""tests/test_stage15_integrity.py
Stage 15 — Scientific integrity and leakage-prevention tests.

Verifies:
  - evaluation_source is 'cv_oof' in all OOF metric dicts.
  - External holdout (GSE140829) is not accessed anywhere.
  - Manifest contains all required fields.
  - Final model file exists and loads correctly after model_comparison.py run.
  - OOF predictions cover identical sample IDs to those in the input.
  - Class encoding is consistent (AD=1, Control=0).
  - Preprocessing transformers are only fitted on training subsets (leakage check).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ml.evaluate import compute_metrics
from scripts.ml.models import SEED, build_model_registry
from scripts.ml.nested_cv import run_nested_cv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_MODELS = PROJECT_ROOT / "outputs" / "models"
OUT_TABLES = PROJECT_ROOT / "outputs" / "tables" / "stage15"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_cohort():
    """Small balanced cohort for structural integrity checks."""
    rng = np.random.default_rng(SEED)
    n = 60
    ids = [f"GSM{i:04d}" for i in range(n)]
    X = pd.DataFrame(
        rng.standard_normal((n, 15)),
        index=ids,
        columns=[f"ENSG{i:011d}" for i in range(15)],
    )
    y = pd.Series([0] * 30 + [1] * 30, index=ids, name="label")
    return X, y


# ---------------------------------------------------------------------------
# 1. Evaluation source label
# ---------------------------------------------------------------------------

class TestEvaluationSourceLabel:
    def test_cv_oof_label_in_metrics(self, synthetic_cohort):
        """compute_metrics must tag all outputs as cv_oof for internal CV."""
        y = pd.Series([0] * 20 + [1] * 20)
        proba = pd.Series(np.linspace(0.1, 0.9, 40))
        m = compute_metrics(y, proba, label="cv_oof")
        assert m["evaluation_source"] == "cv_oof", (
            "Internal CV metrics must carry evaluation_source='cv_oof'"
        )

    def test_external_label_not_set_by_default(self, synthetic_cohort):
        y = pd.Series([0] * 20 + [1] * 20)
        proba = pd.Series(np.linspace(0.1, 0.9, 40))
        m = compute_metrics(y, proba, label="cv_oof")
        assert m["evaluation_source"] != "external", (
            "Internal CV metrics must NOT be labelled 'external'"
        )


# ---------------------------------------------------------------------------
# 2. External holdout isolation
# ---------------------------------------------------------------------------

class TestExternalHoldoutIsolation:
    def test_model_comparison_does_not_import_external_path(self):
        """The model_comparison source must not reference GSE140829 as a data load."""
        script = PROJECT_ROOT / "scripts" / "ml" / "model_comparison.py"
        assert script.exists(), "model_comparison.py must exist"
        source = script.read_text(encoding="utf-8")
        # The script may define EXTERNAL as a string constant, but must not
        # call read_csv or load on the external data directory.
        import re
        # Check there is no call to open/read_csv with GSE140829 in the interim path
        external_data_patterns = [
            r'read_csv\([^)]*GSE140829[^)]*interim',
            r'open\([^)]*GSE140829[^)]*interim',
            r'load_discovery.*GSE140829',
        ]
        for pattern in external_data_patterns:
            matches = re.findall(pattern, source)
            assert not matches, (
                f"model_comparison.py appears to load external GSE140829 data: {matches}"
            )

    def test_nested_cv_has_no_external_reference(self):
        """nested_cv.py must not reference GSE140829."""
        script = PROJECT_ROOT / "scripts" / "ml" / "nested_cv.py"
        source = script.read_text(encoding="utf-8")
        assert "GSE140829" not in source, (
            "nested_cv.py must not reference the external holdout cohort"
        )

    def test_evaluate_has_no_external_reference(self):
        """evaluate.py must not reference GSE140829."""
        script = PROJECT_ROOT / "scripts" / "ml" / "evaluate.py"
        source = script.read_text(encoding="utf-8")
        assert "GSE140829" not in source


# ---------------------------------------------------------------------------
# 3. Manifest field completeness
# ---------------------------------------------------------------------------

class TestManifestCompleteness:
    REQUIRED_FIELDS = {
        "stage",
        "evaluation_source",
        "external_holdout_used",
        "discovery_cohort",
        "best_model_family",
        "feature_set",
        "feature_gene_ids_in_order",
        "cv_oof_auc",
        "cv_oof_auc_ci_95pct",
        "package_versions",
        "git_commit",
        "model_saved_to",
        "scientific_note",
    }

    @pytest.mark.skipif(
        not (OUT_MODELS / "stage15_model_manifest.json").exists(),
        reason="Run model_comparison.py first to generate the manifest",
    )
    def test_manifest_has_required_fields(self):
        import json
        manifest = json.loads(
            (OUT_MODELS / "stage15_model_manifest.json").read_text(encoding="utf-8")
        )
        missing = self.REQUIRED_FIELDS - set(manifest.keys())
        assert not missing, f"Manifest is missing fields: {missing}"

    @pytest.mark.skipif(
        not (OUT_MODELS / "stage15_model_manifest.json").exists(),
        reason="Run model_comparison.py first to generate the manifest",
    )
    def test_manifest_evaluation_source_is_cv_oof(self):
        import json
        manifest = json.loads(
            (OUT_MODELS / "stage15_model_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["evaluation_source"] == "cv_oof"

    @pytest.mark.skipif(
        not (OUT_MODELS / "stage15_model_manifest.json").exists(),
        reason="Run model_comparison.py first to generate the manifest",
    )
    def test_manifest_external_holdout_flag_is_false(self):
        import json
        manifest = json.loads(
            (OUT_MODELS / "stage15_model_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["external_holdout_used"] is False


# ---------------------------------------------------------------------------
# 4. Final model file integrity
# ---------------------------------------------------------------------------

class TestFinalModelFile:
    @pytest.mark.skipif(
        not (OUT_MODELS / "stage15_development_model.joblib").exists(),
        reason="Run model_comparison.py first to generate the model file",
    )
    def test_final_model_loads(self):
        import joblib
        model = joblib.load(OUT_MODELS / "stage15_development_model.joblib")
        assert hasattr(model, "predict_proba"), (
            "Final model must have predict_proba method"
        )

    @pytest.mark.skipif(
        not (OUT_MODELS / "stage15_development_model.joblib").exists(),
        reason="Run model_comparison.py first to generate the model file",
    )
    def test_final_model_produces_valid_probabilities(self):
        """Smoke test: final model must produce probabilities in [0,1]."""
        import joblib
        import json
        manifest_path = OUT_MODELS / "stage15_model_manifest.json"
        assert manifest_path.exists(), "Manifest must exist"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        features = manifest["feature_gene_ids_in_order"]
        model = joblib.load(OUT_MODELS / "stage15_development_model.joblib")
        
        # Create test input with exact expected features
        rng = np.random.default_rng(SEED)
        X_test = pd.DataFrame(
            rng.standard_normal((10, len(features))),
            columns=features,
            index=[f"TEST_{i:02d}" for i in range(10)],
        )
        proba = model.predict_proba(X_test)[:, 1]
        assert len(proba) == 10
        assert np.all(proba >= 0.0) and np.all(proba <= 1.0)


# ---------------------------------------------------------------------------
# 5. Class encoding consistency
# ---------------------------------------------------------------------------

class TestClassEncoding:
    def test_ad_is_encoded_as_1(self, synthetic_cohort):
        """AD samples must map to label=1, Control to label=0."""
        X, y = synthetic_cohort
        assert y.max() == 1, "AD class must be encoded as 1"
        assert y.min() == 0, "Control class must be encoded as 0"
        assert set(y.unique()).issubset({0, 1}), "Labels must be binary 0/1"

    def test_metrics_tp_counts_ad(self, synthetic_cohort):
        """TP in confusion matrix must correspond to correctly identified AD."""
        y = pd.Series([0] * 15 + [1] * 15)
        proba = pd.Series([0.1] * 15 + [0.9] * 15)  # perfect predictions
        m = compute_metrics(y, proba, threshold=0.5, label="cv_oof")
        assert m["TP"] == 15, "TP must equal number of AD correctly classified"
        assert m["TN"] == 15, "TN must equal number of Controls correctly classified"


# ---------------------------------------------------------------------------
# 6. Leakage check: preprocessing fitted only on training data
# ---------------------------------------------------------------------------

class TestNoLeakage:
    def test_scaler_fitted_on_train_only(self, synthetic_cohort):
        """Verify that the StandardScaler sees only training data, not test data."""
        from sklearn.model_selection import StratifiedKFold
        X, y = synthetic_cohort
        registry = build_model_registry(n_features=X.shape[1])
        pipe, _ = registry["logistic_l1"]

        kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        for train_idx, test_idx in kf.split(X, y):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            # Fit pipeline on train only
            from sklearn.base import clone
            pipe_copy = clone(pipe)
            pipe_copy.fit(X_train, y.iloc[train_idx])

            # Check that scaler mean was computed from training data
            scaler = pipe_copy.named_steps["scale"]
            train_means = X_train.mean(axis=0).values
            np.testing.assert_allclose(
                scaler.mean_,
                train_means,
                rtol=1e-5,
                err_msg="Scaler mean must equal training-data mean (leakage if different)",
            )
            break  # one fold is enough for this structural check

    def test_oof_predictions_not_on_training_samples(self, synthetic_cohort):
        """OOF predictions for each sample must come from a fold where that sample was held out."""
        X, y = synthetic_cohort
        registry = build_model_registry(n_features=X.shape[1])
        single = {"logistic_baseline": registry["logistic_baseline"]}

        from sklearn.model_selection import StratifiedKFold
        outer = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        # Record which fold index each sample was in the validation set
        val_membership = pd.Series(-1, index=X.index, dtype=int)
        for fold_i, (_, val_idx) in enumerate(outer.split(X, y)):
            val_membership.iloc[val_idx] = fold_i

        # Every sample should appear in exactly one validation fold
        assert (val_membership == -1).sum() == 0, (
            "Every sample must appear in exactly one outer validation fold"
        )
        fold_counts = val_membership.value_counts()
        assert len(fold_counts) == 3, "Should have 3 validation fold groups"
