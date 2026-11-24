# Stage 14 — ML Interface Contract

## 1. Purpose

This document defines the interface between Stage 14 (ML dataset
engineering and feature selection) and downstream model development.

Stage 14 provides aligned predictors, binary labels, audit metadata,
feature provenance and unfitted sklearn-compatible selector pipelines.

It does not provide a globally fitted selector or a final classifier.

## 2. Dataset builder

Import:

```python
from scripts.ml.data_builder import build_dataset
```

Build a discovery dataset:

```python
data = build_dataset("data", "GSE63061")
```

Build a replication dataset:

```python
replication = build_dataset("data", "GSE63060")
```

The builder returns a dictionary containing:

| Key | Type | Description |
|---|---|---|
| `X_demographic` | pandas DataFrame | Age and Sex predictors |
| `X_lncrna` | pandas DataFrame | Validated lncRNA probe expression |
| `X_combined` | pandas DataFrame | lncRNA expression plus Age and Sex |
| `y` | pandas Series | Binary target: AD=1, Control=0 |
| `metadata` | pandas DataFrame | Aligned sample and audit metadata |
| `feature_manifest` | pandas DataFrame | Probe identifiers and validation evidence |
| `cohort` | str | Source cohort identifier |
| `excluded_diagnoses` | dict | Counts of excluded diagnosis categories |

All returned predictor groups, labels and metadata use the same
sample index and ordering.

## 3. Feature groups

### A. Demographic

```python
X = data["X_demographic"]
```

Columns:

- `age`
- `sex`

### B. lncRNA only

```python
X = data["X_lncrna"]
```

Columns are validated probe IDs, for example:

- `ILMN_1661523`
- `ILMN_1818577`

Actual columns depend on the selected cohort.

### C. Combined

```python
X = data["X_combined"]
```

Contains all validated lncRNA probes followed by Age and Sex.

Do not add sample IDs, diagnosis, cohort identifiers or technical
metadata to these predictor matrices.

## 4. Target and metadata

```python
y = data["y"]
metadata = data["metadata"]
```

Target mapping:

- AD = 1
- Control = 0

MCI and Unknown samples are excluded by the builder.

Metadata retains sample identifiers, diagnosis, age, sex, batch and
available tissue/cohort fields for auditing and future split design.

Metadata must not be concatenated into predictors without a
scientific justification and leakage review.

## 5. Feature manifest

```python
manifest = data["feature_manifest"]
```

The manifest contains:

- `feature_id`: platform probe identifier.
- `gene_id`: validated lncRNA gene identifier.
- `feature_type`: lncRNA_probe.
- `validation_status`: Stage 10 validation status.
- `sequence`: validated probe sequence.
- `platform`: source microarray platform.

Probe-level features must remain distinct. Do not aggregate probes
by gene without a separate scientific justification.

## 6. Unfitted selector pipeline

Import:

```python
from scripts.ml.feature_selection import make_lasso_selector
```

Create an unfitted selector:

```python
selector = make_lasso_selector(
    feature_group="lncrna",
    C=0.1,
)
```

Supported groups:

- `demographic`
- `lncrna`
- `combined`

The returned sklearn Pipeline contains:

1. Preprocessing.
2. L1 logistic regression feature selection.

The pipeline is deliberately unfitted when returned.

## 7. Required cross-validation usage

Feature selection and preprocessing must be fitted independently
inside every training fold.

Example:

```python
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

selector = make_lasso_selector("lncrna", C=0.1)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

for train_idx, test_idx in cv.split(X, y):
    fold_selector = clone(selector)

    # Fit only on the training partition.
    fold_selector.fit(
        X.iloc[train_idx],
        y.iloc[train_idx],
    )

    # The held-out partition must not fit preprocessing
    # or influence feature selection.
```

If verified subject IDs are available, use a group-aware splitter
and pass the subject groups to its `split()` method. Do not assume
that sample IDs establish subject independence.

Do not fit the selector on the complete dataset before cross-validation.

## 8. Stability analysis

Import:

```python
from scripts.ml.feature_selection import selection_stability
```

Example:

```python
stability, fold_results = selection_stability(
    X,
    y,
    selector,
    cv,
)
```

Outputs:

- `stability`: feature selection frequency across folds.
- `fold_results`: training/held-out sample counts and selected
  feature counts for each fold.

Selection frequency measures consistency under the chosen
cross-validation procedure. It does not establish independent
biological replication or clinical validity.

## 9. Important restrictions

- Do not use Stage 11 full-cohort differential-expression results
  as fixed supervised feature preselection.
- Do not access GSE140829 validation labels.
- Do not include sample IDs, diagnosis or cohort identifiers as
  predictors.
- Do not report smoke-test feature counts as cross-validation results.
- Do not claim final classifier accuracy from Stage 14.
- Do not call selected probes validated biomarkers.
- Preserve probe-level identifiers and their provenance.

## 10. Ownership boundary

Stage 14 owns:

- Dataset alignment and eligibility.
- Predictor group construction.
- Feature provenance.
- Unfitted preprocessing and selector factories.
- Exploratory fold-wise feature stability.

Downstream model development owns any later model training and
evaluation under a separately approved validation design.
