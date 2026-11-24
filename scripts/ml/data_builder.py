"""Build leakage-aware ML datasets from validated blood lncRNA probes."""

from pathlib import Path

import pandas as pd


BINARY_LABELS = {"AD": 1, "Control": 0}
VALIDATED_STATUS = "VALIDATED_UNIQUE_LNCRNA"


def _as_bool(series: pd.Series) -> pd.Series:
    """Normalize CSV boolean values."""
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )


def _validated_probe_manifest(mapping: pd.DataFrame) -> pd.DataFrame:
    """Keep only probes passing strict Stage 10 validation."""
    required = {
        "probe_id",
        "mapping_status",
        "validated_gene",
        "exact_genes",
        "within_two_substitutions",
        "sense_exact",
        "conflicting_sequence_or_annotation",
        "sequence",
        "platform",
    }
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(
            f"Mapping file is missing columns: {sorted(missing)}"
        )

    valid = mapping["mapping_status"].eq(VALIDATED_STATUS)
    valid &= mapping["validated_gene"].notna()
    valid &= mapping["exact_genes"].astype(str).eq(
        mapping["validated_gene"].astype(str)
    )
    valid &= mapping["within_two_substitutions"].astype(str).eq(
        mapping["validated_gene"].astype(str)
    )
    valid &= _as_bool(mapping["sense_exact"])
    valid &= ~_as_bool(mapping["conflicting_sequence_or_annotation"])
    valid &= mapping["sequence"].astype(str).str.fullmatch(r"[ACGT]{50}")

    manifest = mapping.loc[valid].copy()

    if manifest["probe_id"].duplicated().any():
        raise ValueError("Validated mapping contains duplicate probe IDs.")

    manifest = manifest.rename(
        columns={
            "probe_id": "feature_id",
            "validated_gene": "gene_id",
        }
    )
    manifest["feature_type"] = "lncRNA_probe"
    manifest["validation_status"] = VALIDATED_STATUS

    return manifest[
        [
            "feature_id",
            "gene_id",
            "feature_type",
            "validation_status",
            "sequence",
            "platform",
        ]
    ].reset_index(drop=True)


def build_dataset(data_root: str | Path, cohort: str = "GSE63061") -> dict:
    """
    Build aligned predictor groups, labels, metadata and feature manifest.

    AD = 1 and Control = 0. MCI and Unknown are excluded.
    No label-informed feature selection happens in this function.
    """
    if cohort not in {"GSE63061", "GSE63060"}:
        raise ValueError("Only GSE63061 and GSE63060 are supported.")

    root = Path(data_root)

    matrix_path = (
        root / "processed" / "stage10"
        / f"{cohort}_validated_lncrna.parquet"
    )
    metadata_path = root / "interim" / cohort / "metadata.csv"
    mapping_path = root / "annotations" / f"{cohort}_probe_mapping.csv"

    for path in (matrix_path, metadata_path, mapping_path):
        if not path.exists():
            raise FileNotFoundError(f"Required input file not found: {path}")

    expression = pd.read_parquet(matrix_path)
    metadata = pd.read_csv(metadata_path)
    mapping = pd.read_csv(mapping_path)

    required_meta = {
        "sample_id", "diagnosis", "age", "sex", "batch"
    }
    missing_meta = required_meta - set(metadata.columns)
    if missing_meta:
        raise ValueError(
            f"Metadata is missing columns: {sorted(missing_meta)}"
        )

    if metadata["sample_id"].duplicated().any():
        raise ValueError("Metadata contains duplicate sample IDs.")

    if expression.index.duplicated().any():
        raise ValueError("Expression matrix contains duplicate sample IDs.")

    if expression.columns.duplicated().any():
        raise ValueError("Expression matrix contains duplicate feature IDs.")

    # Align metadata to expression using explicit sample IDs.
    metadata = metadata.set_index("sample_id", drop=False)

    missing_samples = metadata.index.difference(expression.index)
    extra_samples = expression.index.difference(metadata.index)
    if len(missing_samples) or len(extra_samples):
        raise ValueError(
            "Expression/metadata sample mismatch: "
            f"{len(missing_samples)} missing from expression; "
            f"{len(extra_samples)} missing from metadata."
        )

    metadata = metadata.loc[expression.index].copy()

    # Record excluded diagnosis groups before filtering.
    metadata["diagnosis"] = metadata["diagnosis"].astype(str).str.strip()
    keep = metadata["diagnosis"].isin(BINARY_LABELS)
    excluded_diagnoses = (
        metadata.loc[~keep, "diagnosis"].value_counts().to_dict()
    )
    metadata = metadata.loc[keep].copy()

    y = metadata["diagnosis"].map(BINARY_LABELS).astype("int8")
    y.name = "target"

    # Retain only strictly validated Stage 10 probes.
    manifest = _validated_probe_manifest(mapping)
    feature_ids = manifest["feature_id"].tolist()

    missing_features = sorted(set(feature_ids) - set(expression.columns))
    if missing_features:
        raise ValueError(
            f"{len(missing_features)} validated probes are missing "
            "from the expression matrix."
        )

    X_lncrna = expression.loc[metadata.index, feature_ids].copy()
    X_lncrna = X_lncrna.apply(pd.to_numeric, errors="coerce")

    # Demographic predictors are separate from lncRNA predictors.
    X_demographic = metadata[["age", "sex"]].copy()
    X_demographic["age"] = pd.to_numeric(
        X_demographic["age"], errors="coerce"
    )
    X_demographic["sex"] = (
        X_demographic["sex"].astype("string").str.strip()
    )
    X_demographic["sex"] = X_demographic["sex"].replace("", pd.NA)

    # Combined group contains both lncRNA and demographic predictors.
    X_combined = pd.concat(
        [X_lncrna, X_demographic], axis=1
    )

    # Audit metadata is kept separate and never used automatically as X.
    audit_columns = [
        col for col in [
            "sample_id", "dataset_id", "diagnosis",
            "age", "sex", "batch", "tissue",
        ]
        if col in metadata.columns
    ]
    aligned_metadata = metadata[audit_columns].copy()

    return {
        "X_demographic": X_demographic,
        "X_lncrna": X_lncrna,
        "X_combined": X_combined,
        "y": y,
        "metadata": aligned_metadata,
        "feature_manifest": manifest,
        "cohort": cohort,
        "excluded_diagnoses": excluded_diagnoses,
    }
