# Model Freeze Manifest — Stage 15 Development Model

**Project:** lncSignal-AD  
**Stage:** 15 — Machine Learning and Evaluation  
**Status:** FROZEN FOR EXTERNAL VALIDATION  
**Freeze Date:** 2026-11-22  
**Evaluation Source:** `cv_oof` (Internal Discovery Cohort Nested Cross-Validation)  
**External Cohort Used:** `False` (GSE140829 remains completely sealed)

---

## 1. Specification Overview

This manifest documents the frozen candidate model trained on the development (discovery) cohort GSE63061. It details the preprocessing pipeline, hyperparameters, selected feature space, internal cross-validation performance, software environment, and exact instructions for downstream evaluation by external validation contributors.

| Property | Value |
|---|---|
| **Discovery Cohort** | GSE63061 (Whole Blood, Illumina HumanHT-12 v4) |
| **Cohort Samples** | N = 273 (AD: 139, Control: 134) |
| **Model Family** | L1-regularized Logistic Regression (`logistic_l1`) / Elastic Net |
| **Feature Set** | `lncrna_only` (572 validated lncRNA gene symbols/IDs) |
| **Target Variable** | Binary Diagnosis (AD = 1, Control = 0) |
| **Model File** | `outputs/models/stage15_development_model.joblib` |
| **Machine-Readable Manifest** | `outputs/models/stage15_model_manifest.json` |

---

## 2. Model Pipeline & Preprocessing

The model is packaged as a strict scikit-learn `Pipeline` to prevent any data leakage during external testing:

1. **Imputation:** `SimpleImputer(strategy='median')`
   - Handles missing probe/gene expression values using medians derived strictly from training data.
2. **Standardization:** `StandardScaler(with_mean=True, with_std=True)`
   - Centers and scales features based strictly on training fold distributions.
3. **Classifier:** `LogisticRegression(penalty='l1', solver='saga', max_iter=2000, class_weight='balanced', random_state=42)`
   - Performs embedded L1 sparse feature selection and binary classification.

> **CRITICAL:** Downstream evaluators MUST NOT apply external scaling or imputation before calling `predict_proba`. Feed raw or normalized expression matrix directly into the pipeline object.

---

## 3. Internal Performance Summary (OOF Nested CV)

Evaluated via 5-fold outer, 3-fold inner stratified cross-validation on GSE63061:

- **Metric Source:** `cv_oof` (Out-of-Fold across all 273 discovery subjects)
- **ROC-AUC:** Documented in `outputs/models/stage15_model_manifest.json`
- **Threshold Selection:** Selected via training folds / Youden index on OOF probabilities (never tuned against holdout targets)

---

## 4. Software Dependencies

- `python`: >= 3.10
- `scikit-learn`: >= 1.3.0
- `numpy`: >= 1.24.0
- `pandas`: >= 2.0.0
- `joblib`: >= 1.3.0

---

## 5. Instructions for External Validation (Gate E / Friend 4)

```python
import joblib
import json
import pandas as pd

# 1. Load manifest and verify feature ordering
with open("outputs/models/stage15_model_manifest.json", "r") as f:
    manifest = json.load(f)

expected_features = manifest["feature_gene_ids_in_order"]

# 2. Align external dataset columns to expected_features
# X_external must be a DataFrame with exactly expected_features as columns in order
X_aligned = X_external[expected_features]

# 3. Load model and predict
model = joblib.load("outputs/models/stage15_development_model.joblib")
probabilities = model.predict_proba(X_aligned)[:, 1]

# 4. Binary predictions using frozen threshold
threshold = manifest["threshold_used"]
predictions = (probabilities >= threshold).astype(int)
```
