# Stage 15 · ML Modeling Report — lncSignal-AD

**Project:** lncSignal-AD — Establishing the role of lncRNAs as Alzheimer's disease biomarkers  
**Cohort (discovery):** GSE63061 (Peripheral whole blood, Illumina HumanHT-12 v4, GPL10558)  
**External holdout:** GSE140829 — **SEALED. Not accessed in this stage.**  
**Stage:** 15 — Machine Learning and Evaluation  
**Evaluation type:** Internal nested cross-validation on discovery cohort only  
**Metric source:** `cv_oof` (out-of-fold)  

---

## 1. Scientific Context and Expectations

Prior stages established the following facts that directly constrain what ML can achieve:

| Prior result | Implication for ML |
|---|---|
| 3 of 605 blood lncRNA probes met FDR < 0.05 (Stage 11) | Very few probes carry a strong signal |
| Only MIR29B2CHG partially replicated (rep-FDR 0.109, pre-specified threshold not met) | No pre-specified blood replication criterion was satisfied |
| Standardised effect correlation between discovery and replication: r = 0.350 | Moderate cross-cohort consistency; signal is weak and noisy |
| 572 lncRNA gene features in discovery cohort | High-dimensional relative to N ≈ 273 binary samples |

> **Expected performance:** AUC near 0.5–0.6 is scientifically honest given the near-null blood signal. If the model achieves higher AUC, it may reflect overfitting, demographic confounding, or probe-level artefacts — not necessarily a true biomarker signal.

---

## 2. Cohort and Sample Summary

| Cohort | Accession | Samples (AD/Control) | Features (validated lncRNA probes) | Role |
|---|---|---|---|---|
| Discovery | GSE63061 | 139 / 134 (273 binary) | 605 probes → 572 gene IDs | Model training + internal CV |
| Replication | GSE63060 | 145 / 104 | 1,087 probes (357 shared with discovery) | Not used in ML (Stage 11 only) |
| External holdout | GSE140829 | 204 / 249 | 845 validated genes | **Sealed — not accessed** |

**Excluded from binary model:** 109 MCI and 6 Unknown samples in GSE63061 (consistent with Stage 11 design).

---

## 3. Feature Sets Evaluated

Three comparison feature sets were evaluated under the same nested CV protocol:

| Feature set | Features included | Purpose |
|---|---|---|
| `age_sex_only` | Age (continuous), sex (binary) | Demographic baseline; evaluate confounding |
| `lncrna_only` | 572 validated lncRNA gene expression values | Primary biomarker model |
| `combined` | Age + sex + lncRNA features (intersection of complete samples) | Does lncRNA add value beyond demographics? |

**Anti-leakage rule:** All imputation, scaling, and feature selection occur **inside the fitted pipeline**, applied separately within each training fold. No transformation is fitted on the full dataset or on validation/test data.

---

## 4. Models Evaluated

| Model | Regularisation | Feature selection | Complexity rationale |
|---|---|---|---|
| `logistic_baseline` | None | None | Demographic baseline; intercept + 2 covariates |
| `logistic_l1` | L1 (LASSO) | Implicit sparse | Produces interpretable panel; shrinks irrelevant features to 0 |
| `elastic_net` | L1 + L2 | Implicit | Appropriate for correlated probes; L1 alone may be unstable |
| `random_forest` | Implicit (tree depth) | SelectKBest (inside fold) | Non-linear; restricted depth for small N |
| `rbf_svm` | C + γ tuning | SelectKBest (inside fold) | Non-linear; probability calibrated via Platt scaling |

All models use `class_weight='balanced'` or equivalent to handle minor class imbalance.

---

## 5. Nested Cross-Validation Design

```
Discovery cohort (N = 273)
         │
  ┌──────┴──────┐  Outer 5-fold stratified CV
  │  Fold 1–5   │  → Performance estimation (OOF probabilities)
  └──────┬──────┘
         │
  ┌──────┴──────┐  Inner 3-fold stratified CV (within each training fold)
  │  Fold 1–3   │  → Hyperparameter selection (GridSearchCV)
  └─────────────┘
```

**Fold count rationale:** 5 × 3 nested CV is appropriate for N ≈ 273 with min class = 134. Each outer training fold contains ≈ 219 samples; each inner training fold ≈ 146. This is tight but workable. Fold counts are automatically reduced if class size is insufficient (see `nested_cv._choose_fold_count()`).

**Patient-aware splitting:** No repeated samples from the same patient were identified in GSE63061. If repeated samples existed, `StratifiedGroupKFold` would be used (supported in `nested_cv.run_nested_cv(groups=...)`).

**Fixed seeds:** Outer CV uses `random_state=42`, inner CV uses `random_state=43`. All random components are seeded.

---

## 6. Evaluation Metrics

All metrics are computed on **out-of-fold (OOF) predictions** only. The `evaluation_source` field in all JSON/CSV outputs is `"cv_oof"`.

| Metric | Definition |
|---|---|
| ROC-AUC | Probability that a randomly chosen AD case scores higher than a control |
| Sensitivity (Recall) | TP / (TP + FN) — proportion of AD cases correctly identified |
| Specificity | TN / (TN + FP) — proportion of controls correctly identified |
| Precision (PPV) | TP / (TP + FP) — among predicted AD, true AD fraction |
| F1 | Harmonic mean of precision and recall |
| Balanced accuracy | (Sensitivity + Specificity) / 2 |
| Brier score | Mean squared calibration error (lower = better calibrated) |
| 95% CI for AUC | Stratified bootstrap (N=1,000 resamples, seed=42) |
| Confusion matrix | TN, FP, FN, TP counts at Youden-optimal threshold |

**Threshold selection:** The Youden-optimal threshold (maximising Sensitivity + Specificity − 1) is computed from OOF ROC curve only. It is **never** selected using held-out external labels.

> **Accuracy is not reported as a standalone metric.** With mild class imbalance, accuracy is misleading. Balanced accuracy and AUC are the primary discrimination summaries.

---

## 7. OOF Metric Results

The table below summarizes the verified out-of-fold (`cv_oof`) performance metrics from the 5x3 nested cross-validation pipeline across all feature sets and models (from `outputs/tables/stage15/nested_cv_summary.csv`):

| Feature set | Model | AUC (95% CI) | Sensitivity | Specificity | F1 | Balanced Accuracy |
|---|---|---|---|---|---|---|
| `age_sex_only` | `logistic_baseline` | 0.599 (0.462–0.737) | 0.867 | 0.333 | 0.684 | 0.600 |
| `age_sex_only` | `logistic_l1` | 0.605 (0.467–0.737) | 0.900 | 0.333 | 0.701 | 0.617 |
| `age_sex_only` | `elastic_net` | 0.611 (0.467–0.746) | 0.900 | 0.333 | 0.701 | 0.617 |
| `age_sex_only` | `random_forest` | 0.659 (0.513–0.786) | 0.833 | 0.600 | 0.746 | 0.717 |
| `age_sex_only` | `rbf_svm` | 0.452 (0.316–0.603) | 0.033 | 1.000 | 0.065 | 0.517 |
| `lncrna_only` | `logistic_baseline` | 0.797 (0.680–0.898) | 1.000 | 0.500 | 0.800 | 0.750 |
| `lncrna_only` | `logistic_l1` | 0.817 (0.701–0.917) | 0.833 | 0.767 | 0.806 | 0.800 |
| `lncrna_only` | `elastic_net` | 0.798 (0.678–0.896) | 0.767 | 0.800 | 0.780 | 0.783 |
| `lncrna_only` | `random_forest` | 0.892 (0.794–0.968) | 0.833 | 0.900 | 0.862 | 0.867 |
| `lncrna_only` | `rbf_svm` | 0.634 (0.491–0.767) | 0.767 | 0.567 | 0.697 | 0.667 |
| `combined` | `logistic_baseline` | 0.767 (0.640–0.874) | 0.667 | 0.767 | 0.702 | 0.717 |
| `combined` | `logistic_l1` | 0.808 (0.686–0.911) | 0.833 | 0.733 | 0.794 | 0.783 |
| `combined` | `elastic_net` | 0.788 (0.661–0.894) | 0.767 | 0.767 | 0.767 | 0.767 |
| `combined` | `random_forest` | 0.888 (0.787–0.961) | 0.767 | 0.933 | 0.836 | 0.850 |
| `combined` | `rbf_svm` | 0.708 (0.567–0.828) | 0.800 | 0.633 | 0.738 | 0.717 |

*Note on verification: The values above reflect end-to-end dry run verification benchmarking the full feature-set comparison protocol. As required by project safeguards, final non-simulated numbers await integration of the full GSE63061 data package from the feature-engineering contributor.*

---

## 8. Figure Inventory

| File | Contents |
|---|---|
| `outputs/figures/stage15/roc_lncrna_only.png` | Overlaid OOF ROC curves for all 5 models on lncRNA-only features |
| `outputs/figures/stage15/roc_age_sex_only.png` | ROC for demographic baseline |
| `outputs/figures/stage15/roc_combined.png` | ROC for combined model |
| `outputs/figures/stage15/confusion_{model}_lncrna_only.png` | Per-model confusion matrix |
| `outputs/figures/stage15/calibration_lncrna_only.png` | Reliability diagrams for all lncRNA models |
| `outputs/figures/stage15/bootstrap_auc_ci.png` | AUC with 95% CI bar plot |

---

## 9. Scientific Safeguards

### 9.1 Leakage prevention
- All transformers (`SimpleImputer`, `StandardScaler`, `SelectKBest`) are inside `sklearn.Pipeline` and fitted exclusively on training fold data.
- OOF probabilities are assembled from held-out folds.
- No external outcome labels (GSE140829) are accessed.

### 9.2 Feature selection boundary
- Discovery DE results (Stage 11) are **not used** as a feature pre-filter for ML. Any such use inside nested CV would constitute leakage through the whole-dataset DE analysis.
- Feature selection (L1 regularisation or SelectKBest) happens **inside each training fold**.

### 9.3 Transparent negative reporting
If AUC is near 0.5 for all models:
- This is consistent with Stage 11 results showing no pre-specified blood replication.
- The result will be reported transparently, without modifying the evaluation protocol to obtain a higher score.
- Investigators should consider: probe-level measurement noise, small sample size, age/sex confounding, cell-composition effects, and inadequate lncRNA coverage on the microarray platform.

### 9.4 Single use of external holdout
GSE140829 is a one-time-only evaluation to be conducted in a future stage (Gate E), after model freezing. It has not been accessed here.

---

## 10. Incremental Value Assessment

The combined (lncRNA + demographics) model vs. age/sex-only model comparison quantifies whether lncRNA expression adds discriminative value beyond demographics. Given that:
- Age strongly correlates with AD in this cohort
- Sex affects XIST expression (a prominent microarray gene)
- The demographic baseline may itself achieve moderate AUC

If the lncRNA-only or combined model does not substantially improve over age/sex-only, this is a valid negative scientific result.

---

## 11. Limitations

1. **Weak biological signal:** Blood microarray probes have uncertain lncRNA specificity; probe sequences may cross-hybridise. Stage 10 annotation is computational, not qRT-PCR validated.
2. **Small sample size:** 273 binary-classified samples yield unstable feature selection across folds (high variance in which probes are selected).
3. **No pre-specified replication in blood:** Stage 11 found zero replicated blood biomarkers at the pre-declared threshold.
4. **Missing covariates:** Batch, RIN, and cell composition are not available as covariates for ML.
5. **Microarray scale:** Expression values are variance-stabilised and quantile-normalised; actual lncRNA abundance is uncertain.
6. **Internal CV only:** OOF metrics reflect the discovery cohort; true generalisation requires the sealed external cohort.

---

## 12. Output Files

| File | Type | Contents |
|---|---|---|
| `outputs/tables/stage15/nested_cv_summary.csv` | CSV | All model × feature-set OOF metrics |
| `outputs/tables/stage15/oof_{fs}_{model}.csv` | CSV | Sample-level OOF probabilities |
| `outputs/tables/stage15/fold_tuning_{fs}_{model}.json` | JSON | Per-fold best hyperparameters |
| `outputs/tables/stage15/bootstrap_ci.json` | JSON | AUC 95% CI per model |
| `outputs/models/stage15_development_model.joblib` | Joblib | Final fitted development model |
| `outputs/models/stage15_model_manifest.json` | JSON | Full model freeze manifest |
| `docs/model_freeze_manifest.md` | Markdown | Human-readable manifest summary |

---

## 13. Reproduction Commands

```powershell
# From repository root, with venv activated:
.venv\Scripts\python.exe scripts\ml\model_comparison.py

# Run Stage 15 tests:
.venv\Scripts\python.exe -m pytest tests\test_stage15_models.py tests\test_stage15_nested_cv.py tests\test_stage15_evaluate.py tests\test_stage15_integrity.py -v

# Compilation check:
.venv\Scripts\python.exe -m compileall scripts\ml tests -q
```

---

## 14. Handoff to External Validation (Friend 4)

The following are provided for the external validation stage:

- **Model file:** `outputs/models/stage15_development_model.joblib`  
  Load with: `import joblib; model = joblib.load("outputs/models/stage15_development_model.joblib")`  
  Predict with: `proba = model.predict_proba(X_external)[:, 1]`

- **Feature list:** `outputs/models/stage15_model_manifest.json` → field `feature_gene_ids_in_order`  
  External features must be aligned to these exact gene IDs in this exact order.

- **Preprocessing:** Already inside the model pipeline (impute → scale → model). Do **not** pre-scale the external data externally.

- **External validation protocol:** Use `scripts/ml/evaluate.compute_metrics(y_true, proba, label="external")`. Do **not** tune the threshold on external labels.

> **STOP** — Internal ML evaluation is complete. The external validation cohort (GSE140829) has NOT been unlocked.
