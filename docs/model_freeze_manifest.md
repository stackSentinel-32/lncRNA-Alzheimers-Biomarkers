# Stage 15 | Frozen GSE63061 discovery model

The final lncRNA-only refit used the entire real discovery cohort after nested evaluation. Input order is the 605 validated Stage 14 probe columns. The fitted pipeline accepts a sample-by-probe DataFrame with these exact column IDs. Its positive-class probability refers to AD. The decision threshold is 0.5.

| Property | Frozen value |
|---|---|
| Class counts | 139 AD and 134 Control |
| Model | Random forest, 100 trees, maximum depth 5, minimum leaf size 3, balanced class weights |
| Selection | Training-set F-test, five probes |
| Internal evaluation | ROC-AUC 0.7017 for the nested family-selection procedure |
| Training code commit | `1e940fc8318e2745140ce941ca6c7d69a12b7d8c` |
| Model SHA-256 | `366e111cc59b9c1f0df2a62a6da066a99ff94b1244d8fb0412784d4d4e56baa0` |

### Final estimator inputs

| Probe ID | Stage 10 gene ID |
|---|---|
| ILMN_1855278 | ENSG00000231721 |
| ILMN_1859787 | ENSG00000267394 |
| ILMN_1907042 | ENSG00000203709 |
| ILMN_3200414 | ENSG00000304921 |
| ILMN_3297577 | ENSG00000303192 |

The stored pipeline performs median imputation, removes training-constant features, standardizes, selects probes and applies the forest. Keep the input probe identifiers and order, including columns used before selection. The [machine-readable manifest](../outputs/models/stage15_model_manifest.json) records all 605 input IDs, downstream estimator inputs, parameters, label order, source data hashes, package versions and final tuning scores. [Frozen pipeline](../outputs/models/stage15_development_model.joblib)

External assay compatibility and independent evaluation remain outstanding. No external clinical outcomes entered development.
