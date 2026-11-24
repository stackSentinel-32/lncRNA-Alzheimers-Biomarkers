"""Generate a feature eligibility audit for Stage 14."""

from pathlib import Path
import pandas as pd

from scripts.ml.data_builder import _validated_probe_manifest


def build_eligibility_report(data_root="data", output_dir="outputs/tables"):
    """Audit validated lncRNA probe availability in both blood cohorts."""
    root = Path(data_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    reports = []
    cohort_summaries = []

    for cohort in ["GSE63061", "GSE63060"]:
        matrix = pd.read_parquet(
            root / "processed" / "stage10"
            / f"{cohort}_validated_lncrna.parquet"
        )
        metadata = pd.read_csv(
            root / "interim" / cohort / "metadata.csv"
        )
        mapping = pd.read_csv(
            root / "annotations" / f"{cohort}_probe_mapping.csv"
        )

        manifest = _validated_probe_manifest(mapping)
        manifest["cohort"] = cohort

        present = manifest["feature_id"].isin(matrix.columns)
        manifest["present_in_expression"] = present.values

        missingness = matrix.isna().mean()
        variance = matrix.var(axis=0, skipna=True)

        manifest["missing_fraction"] = (
            manifest["feature_id"].map(missingness)
        )
        manifest["zero_variance"] = (
            manifest["feature_id"].map(variance).fillna(0).eq(0)
        )

        manifest["eligible_for_ml"] = (
            manifest["present_in_expression"]
            & manifest["missing_fraction"].eq(0)
            & ~manifest["zero_variance"]
        )

        reports.append(manifest)

        diagnosis_counts = (
            metadata["diagnosis"].value_counts(dropna=False).to_dict()
        )

        cohort_summaries.append({
            "cohort": cohort,
            "total_samples": len(metadata),
            "AD_samples": diagnosis_counts.get("AD", 0),
            "Control_samples": diagnosis_counts.get("Control", 0),
            "MCI_samples": diagnosis_counts.get("MCI", 0),
            "Unknown_samples": diagnosis_counts.get("Unknown", 0),
            "matrix_features": matrix.shape[1],
            "validated_probes": len(manifest),
            "present_probes": int(
                manifest["present_in_expression"].sum()
            ),
            "eligible_probes": int(manifest["eligible_for_ml"].sum()),
            "missing_expression_values": int(matrix.isna().sum().sum()),
        })

    feature_report = pd.concat(reports, ignore_index=True)
    summary_report = pd.DataFrame(cohort_summaries)

    feature_report.to_csv(
        output / "feature_eligibility_by_cohort.csv", index=False
    )
    summary_report.to_csv(
        output / "feature_eligibility_summary.csv", index=False
    )

    print("COHORT SUMMARY")
    print(summary_report.to_string(index=False))
    print("\nFeature-level report:", output / "feature_eligibility_by_cohort.csv")
    print("Summary report:", output / "feature_eligibility_summary.csv")


if __name__ == "__main__":
    build_eligibility_report()
