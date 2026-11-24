# Stage 15 | Internal prediction in GSE63061 whole blood

The lncRNA model-selection procedure achieved ROC-AUC 0.7017 and PR-AUC 0.6967 on held-out discovery samples. At the prespecified AD probability threshold of 0.5, it identified 87 of 139 AD samples and correctly classified 86 of 134 controls. This study evaluates internal discrimination. Patient-level transportability, clinical calibration and external predictive validity remain unknown.

## Cohort and protocol

The Stage 14 builder aligned 273 binary samples by GSM ID and supplied 605 validated probe intensities, age and sex. Diagnosis coding was AD=1 and Control=0. All three predictor groups used the same samples and outer folds. The contributor's dry run (60 artificial samples) previously occupied the production output paths. The real-data entry point now consumes only the GSE63061 Stage 14 outputs. The former real-data loader aggregated probes to genes, applied whole-cohort filtering and selected complete demographic cases. The corrected pipeline retains probe identities and fits preprocessing inside each training partition.

The discovery GEO header contains unique sample-description codes and array positions for these 273 observations. It supplies no independently verified longitudinal patient key. The 34 physical chips inferred from array-title barcodes served as groups in stratified five-fold outer and three-fold inner CV. All samples on a chip stayed together at both levels. The discovery sample codes, titles, centers and chips were excluded from predictors. Unrecognized repeated donors across chips and center-associated confounding remain possible.

Every candidate fitted the Stage 14 training-fold median and categorical mode imputers, one-hot encoder for sex where applicable, variance filter and standardizer. L1 and elastic-net models learn sparse coefficients during training. Forest and SVM models additionally fitted univariate F-test selection inside the pipeline. SVM sigmoid calibration uses three internal training partitions and has no access to inner validation or outer test observations. The calibration subdivisions themselves are sample-based. No Stage 11 DE ranking or full-cohort supervised shortlist entered the evaluation.

Outer held-out fold sizes were 58, 52, 58, 53, 52 samples in fold order, with 7, 7, 7, 7, 6 chips respectively. Every discovery sample received exactly one held-out score per candidate.

### Candidate settings

The search used deterministic seeds 42 (outer split and stochastic estimators) and 43 (inner split). The logistic baseline used unpenalized logistic regression. L1 evaluated C = 0.05, 0.2 and 1. Elastic net evaluated C = 0.05 and 0.2 with L1 ratios 0.25 and 0.75. Forests used 100 trees, minimum leaf size 3 and maximum depths 3 and 5. RBF SVM evaluated C = 0.1 and 1 with default scale gamma. The forest and SVM compared 5 and 20 predictors capped at the raw predictor count. The grid was fixed before the real-data run.

## Comparison of the 15 prespecified candidates

Each row pools one outer-held-out prediction per sample for one candidate. PR-AUC uses trapezoidal integration of the precision-recall curve. Average precision, bootstrap intervals, Brier scores and unrounded metrics are also in the [full comparison](../outputs/tables/stage15/nested_cv_summary.csv). AD prevalence is 139/273 (0.5092). Confusion counts are in TN, FP, FN, TP order.

| Feature group | Model | ROC-AUC | PR-AUC | Sensitivity | Specificity | F1 | Balanced accuracy | TN / FP / FN / TP |
|---|---|---:|---:|---:|---:|---:|---:|---|
| age_sex_only | logistic_baseline | 0.5970 | 0.5742 | 0.6691 | 0.5448 | 0.6348 | 0.6069 | 73 / 61 / 46 / 93 |
| age_sex_only | logistic_l1 | 0.6147 | 0.5827 | 0.6331 | 0.5896 | 0.6241 | 0.6113 | 79 / 55 / 51 / 88 |
| age_sex_only | elastic_net | 0.6148 | 0.6003 | 0.6763 | 0.5448 | 0.6395 | 0.6105 | 73 / 61 / 45 / 94 |
| age_sex_only | random_forest | 0.5768 | 0.5453 | 0.5755 | 0.5597 | 0.5755 | 0.5676 | 75 / 59 / 59 / 80 |
| age_sex_only | rbf_svm | 0.3983 | 0.4394 | 0.6763 | 0.2239 | 0.5579 | 0.4501 | 30 / 104 / 45 / 94 |
| lncrna_only | logistic_baseline | 0.6186 | 0.6102 | 0.5683 | 0.5672 | 0.5725 | 0.5678 | 76 / 58 / 60 / 79 |
| lncrna_only | logistic_l1 | 0.5899 | 0.5674 | 0.6115 | 0.5373 | 0.5944 | 0.5744 | 72 / 62 / 54 / 85 |
| lncrna_only | elastic_net | 0.6098 | 0.5897 | 0.6259 | 0.5373 | 0.6042 | 0.5816 | 72 / 62 / 52 / 87 |
| lncrna_only | random_forest | 0.7052 | 0.6926 | 0.6115 | 0.6418 | 0.6250 | 0.6267 | 86 / 48 / 54 / 85 |
| lncrna_only | rbf_svm | 0.6693 | 0.6444 | 0.6403 | 0.6418 | 0.6449 | 0.6410 | 86 / 48 / 50 / 89 |
| combined | logistic_baseline | 0.6192 | 0.6083 | 0.5683 | 0.5821 | 0.5766 | 0.5752 | 78 / 56 / 60 / 79 |
| combined | logistic_l1 | 0.5844 | 0.5649 | 0.6115 | 0.5224 | 0.5903 | 0.5669 | 70 / 64 / 54 / 85 |
| combined | elastic_net | 0.6280 | 0.6062 | 0.6331 | 0.5597 | 0.6154 | 0.5964 | 75 / 59 / 51 / 88 |
| combined | random_forest | 0.6984 | 0.6848 | 0.5899 | 0.6194 | 0.6029 | 0.6047 | 83 / 51 / 57 / 82 |
| combined | rbf_svm | 0.6626 | 0.6327 | 0.6403 | 0.6119 | 0.6357 | 0.6261 | 82 / 52 / 50 / 89 |

## Evaluation of the selection procedure

The prespecified development target was lncRNA-only prediction. Demographic and combined models were comparators. For each outer training set, rank all five lncRNA families and settings by mean inner ROC-AUC. Break exact family ties in registry order and parameter ties in grid order. Use the winner's refit for that fold's held-out samples. One fold selected SVM and four selected random forest. The full-data training run later applied this same rule. Choosing the highest *observed outer* score across the comparison table would introduce selection optimism.

| ROC-AUC | PR-AUC | Average precision | Sensitivity | Specificity | F1 | Balanced accuracy | TN / FP / FN / TP |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.7017 | 0.6967 | 0.6988 | 0.6259 | 0.6418 | 0.6350 | 0.6338 | 86 / 48 / 52 / 87 |

The 95% percentile interval for procedure ROC-AUC was 0.6391 to 0.7601, from 1,000 chip-level resamples of the fixed out-of-fold predictions. It describes uncertainty conditional on these fitted folds and does not include repeated training or split-selection variability. The five-probe full-data model has no separate independent performance estimate. Association with diagnosis does not establish regulation, causal significance or clinical utility. [Fold choices and exact procedure metrics](../outputs/tables/stage15/selection_procedure.json)

## Reproduction

Run from the project root in the existing Python environment. The analysis reads discovery inputs only. It regenerates the per-sample OOF scores, fold-specific tuning logs, metrics, ROC curves, lncRNA confusion matrices and calibration panels, chip-bootstrap comparison figure and frozen model.

```powershell
.\.venv\Scripts\python.exe -m scripts.ml.model_comparison
.\.venv\Scripts\python.exe -m pytest tests -q --tb=short -rs
```

See the [freeze record](model_freeze_manifest.md) for the full-data refit and input contract.

## Verification

The complete repository suite passed 453 tests with the optional Stage 13 integration checks enabled. Two warnings came from a synthetic small-class test exercising reduced fold counts. The real run used five outer and three inner folds throughout. Verification reproduced the frozen estimator from its recorded parameters, checked the chip partitions and training-only supervised fits, and independently compared the saved predictions, confusion counts, ROC-AUC and PR-AUC for all 15 candidates.
