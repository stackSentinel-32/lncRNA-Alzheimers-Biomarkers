# Stage 14 — ML Feature Engineering and Stable lncRNA Panel

## 1. Objective

Stage 14 prepares reproducible, leakage-aware machine-learning inputs
from validated blood lncRNA expression data.

The task ends at dataset engineering, preprocessing and exploratory
feature-selection stability. It does not train or evaluate a final
classifier and does not establish clinical biomarkers.

## 2. Data sources

Two blood microarray cohorts are used:

- GSE63061: discovery cohort, platform GPL10558.
- GSE63060: replication cohort, platform GPL6947.

Inputs are the shared Stage 10 validated expression matrices,
sample metadata and probe-mapping annotations.

GSE140829 is a locked external validation cohort. Its expression data
and diagnosis labels are not accessed by this Stage 14 workflow.

GSE125583 is a brain RNA-seq dataset and is not used as a blood
predictor cohort.

## 3. Diagnosis labels and sample eligibility

The binary target is:

- AD = 1
- Control = 0

MCI and Unknown samples are excluded. Original diagnosis labels are
not rewritten in the source metadata.

Observed discovery cohort counts:

| Diagnosis | Samples |
|---|---:|
| AD | 139 |
| Control | 134 |
| MCI excluded | 109 |
| Unknown excluded | 6 |
| Total | 388 |

Observed replication cohort counts:

| Diagnosis | Samples |
|---|---:|
| AD | 145 |
| Control | 104 |
| MCI excluded | 80 |
| Unknown excluded | 0 |
| Total | 329 |

The discovery binary dataset contains 273 samples. The replication
binary dataset contains 249 samples.

Expression rows and metadata are aligned using explicit sample IDs.
Duplicate IDs and mismatched sample sets cause errors.

## 4. Probe eligibility and feature provenance

Only probes with mapping status
`VALIDATED_UNIQUE_LNCRNA` are retained.

The validation gate additionally requires:

- A non-missing validated gene identifier.
- Agreement between exact-gene and validated-gene annotations.
- Agreement between within-two-substitution and validated-gene
  annotations.
- Exact-sense sequence evidence.
- No conflicting sequence or annotation.
- A valid 50-base A/C/G/T sequence.

Probe-level features are preserved. Multiple probes are not averaged
into one gene-level measurement because within-gene probe correlations
were previously found to be low in the Stage 10 assessment.

Observed Stage 14 feature eligibility:

| Metric | GSE63061 | GSE63060 |
|---|---:|---:|
| Expression matrix samples | 388 | 329 |
| Matrix features | 605 | 1,125 |
| Strictly validated probes | 605 | 1,125 |
| Validated probes present | 605 | 1,125 |
| Eligible probes | 605 | 1,125 |
| Missing expression values | 0 | 0 |

Eligibility means that the validated probe is present, has no missing
expression values and is not zero-variance in the supplied matrix.
It does not establish biological or clinical validity.

Detailed outputs:

- `outputs/tables/feature_eligibility_summary.csv`
- `outputs/tables/feature_eligibility_by_cohort.csv`

## 5. Predictor groups

Three independent predictor groups are provided.

| Group | Predictors | Discovery shape |
|---|---|---:|
| A — Demographic | Age, Sex | 273 × 2 |
| B — lncRNA | Validated lncRNA probes | 273 × 605 |
| C — Combined | lncRNA probes, Age, Sex | 273 × 607 |

Sample identifiers, diagnosis, dataset identifiers, tissue and batch
are kept in aligned audit metadata, not automatically included as
predictors.

Age is numeric. Sex is categorical and is encoded within preprocessing.

## 6. Preprocessing

The sklearn-compatible preprocessing pipeline contains:

1. Median imputation for numeric predictors.
2. Most-frequent imputation for categorical predictors.
3. One-hot encoding for categorical predictors.
4. Zero-variance filtering.
5. Standard scaling.

All data-dependent preprocessing steps are fitted through the
pipeline's `fit()` method. They must be fitted only on training
partitions during cross-validation.

The preprocessing factory is in
`scripts/ml/preprocessing.py`.

## 7. Feature selection

The selector uses L1-regularized logistic regression through
`SelectFromModel`.

Configuration:

- Solver: `liblinear`
- Penalty: L1
- Class weighting: balanced
- Maximum iterations: 5000
- Default regularization parameter: C = 0.1
- Selection threshold: mean absolute coefficient importance

Smaller C values apply stronger regularization.

The unfitted selector factory is in
`scripts/ml/feature_selection.py`.

No full-cohort Stage 11 differential-expression results are used
as fixed supervised preselection. Stage 11 results remain available
for biological interpretation only.

## 8. Exploratory feature-selection stability

A five-fold stratified analysis was run on GSE63061 using the
lncRNA-only predictor group.

| Fold | Training samples | Held-out samples | Selected features |
|---|---:|---:|---:|
| 1 | 218 | 55 | 39 |
| 2 | 218 | 55 | 39 |
| 3 | 218 | 55 | 40 |
| 4 | 219 | 54 | 36 |
| 5 | 219 | 54 | 42 |

Across folds:

- 109 distinct probes were selected at least once.
- 5 probes were selected in all five folds.
- 36–42 probes were selected in each fold.

These are exploratory selection-stability observations, not
independent validation or proof of biomarker significance.

The five-fold analysis uses stratified splitting. Subject-level
independence has not yet been verified, so group-aware splitting
cannot currently be claimed.

Full-data smoke tests also confirmed that the three selector
configurations fit successfully. Their selected-feature counts are
not cross-validation results and are not used as a final panel.

## 9. Leakage controls

The workflow follows these rules:

- Label-informed feature selection is fitted inside training folds.
- Preprocessing is part of the fitted sklearn pipeline.
- Held-out fold samples are not used to fit preprocessing or selection.
- Sample IDs and diagnosis-derived fields are not predictors.
- Batch and cohort identifiers are not automatically predictors.
- MCI and Unknown are excluded from the binary target.
- No external validation labels are accessed.
- No final classifier or predictive performance is reported.

For future grouped validation, subject identifiers must be verified
and supplied to a compatible group-aware cross-validation splitter.

## 10. Data limitations

The supplied shared dataset package does not contain the original
raw GEO series-matrix files or the Stage 09 sample-level QC flags.

Consequently, this implementation does not independently reconstruct
raw sample descriptions or reapply unavailable Stage 09 QC flags.
This limitation must be considered when interpreting the eligible
sample counts.

The available metadata contains age, sex, batch and diagnosis, but
subject-level independence across samples has not been established.

## 11. Deliverables

- `scripts/ml/data_builder.py`
- `scripts/ml/preprocessing.py`
- `scripts/ml/feature_selection.py`
- `scripts/ml/build_eligibility_report.py`
- `docs/ml_interface.md`
- `docs/ml_feature_engineering.md`
- `tests/test_ml_feature_engineering.py`

Stage 14 stops at the reproducible ML data interface and exploratory
feature-selection stability analysis.
