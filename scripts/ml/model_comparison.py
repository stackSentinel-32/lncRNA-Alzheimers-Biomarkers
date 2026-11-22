"""scripts/ml/model_comparison.py
Stage 15 — Orchestrator for lncSignal-AD ML model training and evaluation.

What this script does:
  1. Loads discovery cohort (GSE63061) via the Stage 10/11 interface.
  2. Builds three feature sets:
       a) age_sex_only      — demographic baseline (no lncRNA features)
       b) lncrna_only       — validated lncRNA probes only
       c) combined          — age/sex + lncRNA features
  3. Runs nested CV for all 5 models × 3 feature sets.
  4. Computes full metrics on each model's out-of-fold predictions.
  5. Selects the best model on 'lncrna_only' by OOF AUC.
  6. Fits a final development model on ALL discovery data (separate from CV).
  7. Saves all outputs under outputs/tables/stage15/, outputs/figures/stage15/,
     and outputs/models/.

What this script does NOT do:
  - Load or use GSE140829 (external holdout) labels.
  - Modify or override evaluation protocol based on observed scores.
  - Report any metric as "external validation".

All metrics are labelled evaluation_source="cv_oof".

Usage (PowerShell, from repo root):
  .venv\\Scripts\\python.exe scripts\\ml\\model_comparison.py

Prerequisites:
  - data/interim/GSE63061/expression.csv      (from Stage 08)
  - data/interim/GSE63061/metadata.csv        (from Stage 08)
  - data/processed/GSE63061_validated_probe_map.csv  (from Stage 10)

The script stops with a clear error if any prerequisite is missing.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure project root is importable (matches conftest.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ml.evaluate import (
    compute_metrics,
    plot_bootstrap_ci,
    plot_calibration,
    plot_confusion_matrix,
    plot_roc_comparison,
)
from scripts.ml.models import SEED, build_model_registry
from scripts.ml.nested_cv import NestedCVResult, fit_final_model, run_nested_cv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUT_TABLES = PROJECT_ROOT / "outputs" / "tables" / "stage15"
OUT_FIGS = PROJECT_ROOT / "outputs" / "figures" / "stage15"
OUT_MODELS = PROJECT_ROOT / "outputs" / "models"

for _p in [OUT_TABLES, OUT_FIGS, OUT_MODELS]:
    _p.mkdir(parents=True, exist_ok=True)

DISCOVERY = "GSE63061"
# External holdout is SEALED — not loaded or referenced in this stage.
EXTERNAL = "GSE140829"


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_expression_and_metadata(accession: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load expression matrix and metadata for a GEO accession.

    Returns:
        (expression_probes_x_samples, metadata_indexed_by_sample_id)
    """
    base = DATA_INTERIM / accession
    exp_path = base / "expression.csv"
    meta_path = base / "metadata.csv"
    for p in [exp_path, meta_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Required file not found: {p}\n"
                "Run Stage 08 (scripts/01_load_geo_datasets.py) first."
            )
    exp = pd.read_csv(exp_path, index_col=0).apply(pd.to_numeric, errors="coerce")
    exp.index = exp.index.astype(str)
    exp.columns = exp.columns.astype(str)
    meta = pd.read_csv(meta_path, dtype={"sample_id": str})
    meta = meta.drop_duplicates(subset=["sample_id"]).set_index("sample_id")
    return exp, meta


def _load_validated_probe_map(accession: str) -> pd.DataFrame:
    """Load the Stage 10 validated probe map; raises if missing or provisional."""
    map_path = DATA_PROCESSED / f"{accession}_validated_probe_map.csv"
    if not map_path.exists():
        raise FileNotFoundError(
            f"Validated probe map not found: {map_path}\n"
            "Run Stage 10 annotation scripts first. Do NOT fabricate this file."
        )
    mp = pd.read_csv(map_path, dtype=str)
    required = {"feature_id", "gene_id", "gene_symbol", "mapping_evidence"}
    if not required.issubset(mp.columns):
        raise ValueError(
            f"Probe map must contain columns {required}. Found: {mp.columns.tolist()}"
        )
    mp = mp.dropna(subset=list(required))
    provisional = mp["mapping_evidence"].str.contains(
        "PROVISIONAL|NOT verified|PLACEHOLDER|SYMBOL_ONLY",
        case=False,
        regex=True,
        na=False,
    )
    if provisional.any():
        raise ValueError(
            f"{provisional.sum()} probe(s) have provisional/placeholder mapping evidence. "
            "Resolve all probe annotations before running ML. Do NOT use unvalidated probes."
        )
    return mp


def load_discovery_cohort(synthetic: bool = False) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load and prepare the GSE63061 discovery cohort.

    Parameters:
        synthetic: If True, generate a synthetic cohort for pipeline testing / dry-run.

    Returns:
        X_lncrna: Samples × gene_id feature matrix (lncRNA gene-level, one row/sample)
        y: Binary labels (1=AD, 0=Control)
        meta: Metadata indexed by sample_id
    """
    if synthetic:
        print(f"[model_comparison] [SIMULATION DRY-RUN] Generating synthetic cohort for pipeline verification ...")
        from sklearn.datasets import make_classification
        rng = np.random.default_rng(SEED)
        n_samples = 60
        n_features = 20
        X_arr, y_arr = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=6,
            n_redundant=2,
            n_classes=2,
            random_state=SEED,
            weights=[0.5, 0.5],
        )
        sample_ids = [f"GSM_SYN_{i:04d}" for i in range(n_samples)]
        gene_ids = [f"ENSG{i:011d}" for i in range(n_features)]
        X_lncrna = pd.DataFrame(X_arr, index=sample_ids, columns=gene_ids)
        y = pd.Series(y_arr, index=sample_ids, name="diagnosis")
        meta = pd.DataFrame({
            "diagnosis": ["AD" if val == 1 else "Control" for val in y_arr],
            "age": rng.integers(60, 85, size=n_samples),
            "sex": rng.choice(["M", "F"], size=n_samples),
        }, index=sample_ids)
        print(
            f"[model_comparison] [SIMULATION] {n_samples} samples "
            f"(AD={int(y.sum())}, Control={int((y==0).sum())}), "
            f"{n_features} features."
        )
        return X_lncrna, y, meta

    print(f"[model_comparison] Loading {DISCOVERY} discovery cohort ...")
    exp, meta = _load_expression_and_metadata(DISCOVERY)
    mp = _load_validated_probe_map(DISCOVERY)

    # Restrict to AD/Control only; keep MCI separate (not included in binary model).
    meta = meta[meta["diagnosis"].isin(["AD", "Control"])].copy()
    if meta["diagnosis"].nunique() < 2:
        raise ValueError("Only one class found in metadata; check label parsing.")
    if len(meta) < 40:
        warnings.warn(
            f"Small binary cohort: {len(meta)} samples. "
            "Nested CV estimates may be unstable.",
            stacklevel=2,
        )

    # Map probes → gene IDs; aggregate multi-probe genes by median.
    mp = mp[mp["feature_id"].isin(exp.index)].copy()
    if mp.empty:
        raise ValueError(
            "No validated probe IDs match the expression matrix columns. "
            "Check that expression.csv and the probe map use the same probe IDs."
        )
    mp = mp.drop_duplicates(subset=["feature_id"])

    selected = exp.loc[mp["feature_id"]].copy()
    selected.index = mp.set_index("feature_id").loc[selected.index, "gene_id"].values
    # Aggregate multiple probes per gene by median (pre-declared).
    X_genes_probes_x_samples = selected.groupby(level=0).median()
    # Transpose to samples × genes.
    X_lncrna = X_genes_probes_x_samples.T
    X_lncrna = X_lncrna.loc[X_lncrna.index.intersection(meta.index)]
    meta = meta.loc[X_lncrna.index]

    # Basic eligibility filters: drop constant and high-missingness features.
    X_lncrna = X_lncrna.replace([np.inf, -np.inf], np.nan)
    X_lncrna = X_lncrna.loc[:, X_lncrna.notna().mean(axis=0) >= 0.80]
    X_lncrna = X_lncrna.loc[:, X_lncrna.nunique(dropna=True) > 1]

    if X_lncrna.shape[1] < 3:
        raise ValueError(
            f"Only {X_lncrna.shape[1]} usable lncRNA features after eligibility filtering. "
            "Check probe map quality and expression file."
        )

    y = (meta["diagnosis"] == "AD").astype(int)
    print(
        f"[model_comparison] {DISCOVERY}: {len(meta)} samples "
        f"(AD={int(y.sum())}, Control={int((y==0).sum())}), "
        f"{X_lncrna.shape[1]} lncRNA features."
    )
    return X_lncrna, y, meta


# ---------------------------------------------------------------------------
# Feature-set builders
# ---------------------------------------------------------------------------

def build_feature_sets(
    X_lncrna: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Build the three feature sets for model comparison.

    Returns:
        dict[feature_set_name -> (X, y)] — only samples with complete
        feature data are included in age_sex_only and combined sets.
    """
    feature_sets: dict[str, tuple[pd.DataFrame, pd.Series]] = {}

    # --- age_sex_only baseline ---
    demo = pd.DataFrame(index=meta.index)
    demo["age"] = pd.to_numeric(meta.get("age", pd.Series(index=meta.index)), errors="coerce")
    sex_raw = meta.get("sex", pd.Series(index=meta.index, dtype=str))
    demo["female"] = (
        sex_raw.astype(str).str.lower().str.strip()
        .map({"female": 1.0, "f": 1.0, "woman": 1.0, "male": 0.0, "m": 0.0, "man": 0.0})
    )
    demo_ok = demo.notna().mean()
    if demo_ok.min() >= 0.60:
        demo_full = demo.dropna(subset=demo.columns[demo.notna().mean() >= 0.60].tolist())
        common_idx = demo_full.index.intersection(y.index)
        if len(common_idx) >= 20 and y.loc[common_idx].nunique() == 2:
            feature_sets["age_sex_only"] = (demo_full.loc[common_idx], y.loc[common_idx])
            print(
                f"[model_comparison] age_sex_only: {len(common_idx)} samples, "
                f"{demo_full.shape[1]} features (age/sex completeness: {demo_ok.to_dict()})"
            )
        else:
            warnings.warn(
                "age_sex_only feature set skipped: insufficient complete cases or only one class.",
                stacklevel=2,
            )
    else:
        warnings.warn(
            f"age/sex metadata coverage too low ({demo_ok.to_dict()}); "
            "age_sex_only baseline will be skipped.",
            stacklevel=2,
        )

    # --- lncrna_only ---
    feature_sets["lncrna_only"] = (X_lncrna, y)
    print(
        f"[model_comparison] lncrna_only: {len(X_lncrna)} samples, "
        f"{X_lncrna.shape[1]} features."
    )

    # --- combined: lncRNA + demographics (intersection of complete samples) ---
    if "age_sex_only" in feature_sets:
        X_demo, y_demo = feature_sets["age_sex_only"]
        common_idx = X_lncrna.index.intersection(X_demo.index)
        if len(common_idx) >= 20 and y.loc[common_idx].nunique() == 2:
            X_comb = pd.concat(
                [X_lncrna.loc[common_idx], X_demo.loc[common_idx]], axis=1
            )
            feature_sets["combined"] = (X_comb, y.loc[common_idx])
            print(
                f"[model_comparison] combined: {len(common_idx)} samples, "
                f"{X_comb.shape[1]} features."
            )
        else:
            warnings.warn(
                "combined feature set skipped: insufficient overlapping samples.",
                stacklevel=2,
            )

    return feature_sets


# ---------------------------------------------------------------------------
# Git commit hash helper
# ---------------------------------------------------------------------------

def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _package_versions() -> dict[str, str]:
    packages = ["scikit-learn", "numpy", "pandas", "joblib", "scipy", "statsmodels"]
    versions: dict[str, str] = {}
    for pkg in packages:
        try:
            import importlib.metadata
            versions[pkg] = importlib.metadata.version(pkg)
        except Exception:
            versions[pkg] = "unknown"
    return versions


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run(synthetic: bool = False) -> None:
    """Main entry point for Stage 15 ML comparison."""
    print("\n" + "=" * 70)
    print("Stage 15 — ML Model Training, Nested CV & Performance Evaluation")
    print("lncSignal-AD project | Discovery cohort: GSE63061")
    print("EXTERNAL HOLDOUT (GSE140829) REMAINS SEALED")
    if synthetic:
        print("MODE: SYNTHETIC DRY-RUN FOR PIPELINE VERIFICATION")
    print("=" * 70 + "\n")

    # 1. Load data
    X_lncrna, y, meta = load_discovery_cohort(synthetic=synthetic)

    # 2. Build feature sets
    feature_sets = build_feature_sets(X_lncrna, y, meta)
    if not feature_sets:
        raise RuntimeError("No usable feature sets could be constructed.")

    # 3. Run nested CV for each feature set × all 5 models
    all_cv_results: dict[str, dict[str, NestedCVResult]] = {}
    all_metrics: list[dict] = []

    for fs_name, (X_fs, y_fs) in feature_sets.items():
        print(f"\n--- Feature set: {fs_name} ({X_fs.shape[1]} features) ---")
        registry = build_model_registry(n_features=X_fs.shape[1])
        cv_results = run_nested_cv(
            X=X_fs,
            y=y_fs,
            models=registry,
            feature_set=fs_name,
        )
        all_cv_results[fs_name] = cv_results

        # Save OOF predictions per model
        for model_name, ncv_result in cv_results.items():
            oof_df = pd.DataFrame({
                "sample_id": ncv_result.oof_probabilities.index,
                "true_label": y_fs.loc[ncv_result.oof_probabilities.index].values,
                "oof_probability_ad": ncv_result.oof_probabilities.values,
                "evaluation_source": "cv_oof",
            })
            oof_df.to_csv(
                OUT_TABLES / f"oof_{fs_name}_{model_name}.csv", index=False
            )

            # Compute metrics
            metrics = compute_metrics(
                y_true=y_fs.loc[ncv_result.oof_probabilities.index],
                proba=ncv_result.oof_probabilities,
                label="cv_oof",
            )
            metrics["model"] = model_name
            metrics["feature_set"] = fs_name
            metrics["n_outer_folds"] = ncv_result.n_outer_folds
            metrics["n_inner_folds"] = ncv_result.n_inner_folds
            all_metrics.append(metrics)

            # Per-model fold tuning log
            fold_log_path = OUT_TABLES / f"fold_tuning_{fs_name}_{model_name}.json"
            fold_log_path.write_text(
                json.dumps(ncv_result.fold_best_params, indent=2), encoding="utf-8"
            )

    # 4. Summary table
    summary_df = pd.DataFrame(all_metrics)
    summary_df = summary_df.sort_values(
        ["feature_set", "auc"], ascending=[True, False]
    ).reset_index(drop=True)
    summary_df.to_csv(OUT_TABLES / "nested_cv_summary.csv", index=False)
    print("\n[model_comparison] Nested CV summary:")
    print(
        summary_df[["feature_set", "model", "auc", "auc_ci_lo", "auc_ci_hi",
                     "sensitivity", "specificity", "f1", "balanced_accuracy"]]
        .to_string(index=False)
    )

    # 5. Figures
    # ROC comparison per feature set
    for fs_name, cv_results in all_cv_results.items():
        y_fs = feature_sets[fs_name][1]
        oof_probs_dict = {
            name: r.oof_probabilities for name, r in cv_results.items()
        }
        fs_metrics = {
            name: {r["auc_ci_lo"]: r["auc_ci_lo"], **r}
            for r in all_metrics
            if r["feature_set"] == fs_name
            for name in [r["model"]]
        }
        # Rebuild as model_name -> metric_dict
        fs_metrics_clean: dict[str, dict] = {
            r["model"]: r
            for r in all_metrics
            if r["feature_set"] == fs_name
        }
        plot_roc_comparison(
            results=fs_metrics_clean,
            y_true=y_fs,
            oof_probs=oof_probs_dict,
            title=f"Discovery cohort nested-CV ROC — {fs_name}",
            save_path=OUT_FIGS / f"roc_{fs_name}.png",
        )

    # Confusion matrices for lncrna_only models
    if "lncrna_only" in all_cv_results:
        y_lnc = feature_sets["lncrna_only"][1]
        for model_name, ncv_result in all_cv_results["lncrna_only"].items():
            m = next(
                r for r in all_metrics
                if r["model"] == model_name and r["feature_set"] == "lncrna_only"
            )
            plot_confusion_matrix(
                y_true=y_lnc,
                proba=ncv_result.oof_probabilities,
                threshold=m["threshold_used"],
                model_name=model_name,
                feature_set="lncrna_only",
                save_path=OUT_FIGS / f"confusion_{model_name}_lncrna_only.png",
            )

    # Calibration for lncrna_only
    if "lncrna_only" in all_cv_results:
        y_lnc = feature_sets["lncrna_only"][1]
        oof_probs_lnc = {
            name: r.oof_probabilities
            for name, r in all_cv_results["lncrna_only"].items()
        }
        plot_calibration(
            y_true=y_lnc,
            oof_probs=oof_probs_lnc,
            save_path=OUT_FIGS / "calibration_lncrna_only.png",
        )

    # Bootstrap CI bar plot
    plot_bootstrap_ci(
        summary_df=summary_df,
        save_path=OUT_FIGS / "bootstrap_auc_ci.png",
    )

    # 6. Bootstrap CI JSON
    ci_data = {
        f"{r['feature_set']}__{r['model']}": {
            "auc": r["auc"],
            "auc_ci_lo": r["auc_ci_lo"],
            "auc_ci_hi": r["auc_ci_hi"],
            "evaluation_source": "cv_oof",
        }
        for r in all_metrics
    }
    (OUT_TABLES / "bootstrap_ci.json").write_text(
        json.dumps(ci_data, indent=2), encoding="utf-8"
    )

    # 7. Select best model family by lncrna_only AUC
    lncrna_rows = summary_df[summary_df["feature_set"] == "lncrna_only"].copy()
    if lncrna_rows.empty:
        # Fallback: use whatever feature set is available
        lncrna_rows = summary_df.copy()
    best_row = lncrna_rows.sort_values("auc", ascending=False).iloc[0]
    best_model_name = best_row["model"]
    best_fs_name = best_row["feature_set"]
    print(
        f"\n[model_comparison] Best model by OOF AUC: {best_model_name} "
        f"(feature_set={best_fs_name}, AUC={best_row['auc']:.4f})"
    )

    # 8. Fit final development model on ALL discovery data
    print("[model_comparison] Fitting final development model on all discovery data ...")
    X_best, y_best = feature_sets[best_fs_name]
    registry_best = build_model_registry(n_features=X_best.shape[1])
    best_pipeline, best_grid = registry_best[best_model_name]
    final_model = fit_final_model(
        X=X_best, y=y_best,
        pipeline=best_pipeline,
        param_grid=best_grid,
        inner_folds=5,
    )

    # Save final model
    model_path = OUT_MODELS / "stage15_development_model.joblib"
    joblib.dump(final_model, model_path)
    print(f"[model_comparison] Final model saved: {model_path}")

    # Feature importance / LASSO coefficients (if applicable)
    if hasattr(final_model.named_steps.get("model", None), "coef_"):
        coef = final_model.named_steps["model"].coef_[0]
        feature_names = X_best.columns.tolist()
        coef_df = pd.DataFrame({"gene_id": feature_names, "coefficient": coef})
        coef_df = coef_df.reindex(
            coef_df["coefficient"].abs().sort_values(ascending=False).index
        )
        nonzero = coef_df[coef_df["coefficient"] != 0]
        nonzero.to_csv(OUT_TABLES / "final_model_nonzero_coefficients.csv", index=False)
        print(f"[model_comparison] Non-zero coefficients: {len(nonzero)}")

    # 9. Model freeze manifest
    commit = _git_commit()
    pkg_versions = _package_versions()

    manifest = {
        "stage": "15",
        "project": "lncSignal-AD",
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "git_commit": commit,
        "evaluation_source": "cv_oof",
        "external_holdout_used": False,
        "discovery_cohort": f"{DISCOVERY}_SYNTHETIC" if synthetic else DISCOVERY,
        "synthetic_dry_run": synthetic,
        "n_samples": int(len(X_best)),
        "n_ad": int(y_best.sum()),
        "n_control": int((y_best == 0).sum()),
        "best_model_family": best_model_name,
        "feature_set": best_fs_name,
        "n_features_in_model": int(X_best.shape[1]),
        "feature_gene_ids_in_order": X_best.columns.tolist(),
        "cv_oof_auc": float(best_row["auc"]),
        "cv_oof_auc_ci_95pct": [float(best_row["auc_ci_lo"]), float(best_row["auc_ci_hi"])],
        "model_saved_to": str(model_path.relative_to(PROJECT_ROOT)),
        "package_versions": pkg_versions,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "scientific_note": (
            "[SIMULATION / DRY-RUN ONLY] Pipeline and interface verification complete. "
            "Real discovery data package integration pending from feature-engineering contributor. "
            "External holdout (GSE140829) has NOT been accessed."
            if synthetic else
            "Blood discovery found only 3 FDR-significant probes with no prespecified "
            "replication. OOF AUC near 0.5 is a scientifically honest expected outcome. "
            "This is a research development model; do not interpret as clinical software. "
            "External holdout (GSE140829) has NOT been accessed."
        ),
        "do_not_use_for": "clinical_diagnosis",
    }

    manifest_json_path = OUT_MODELS / "stage15_model_manifest.json"
    manifest_json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[model_comparison] Manifest saved: {manifest_json_path}")

    print("\n" + "=" * 70)
    print("Stage 15 complete.")
    print(f"  Tables  : {OUT_TABLES}")
    print(f"  Figures : {OUT_FIGS}")
    print(f"  Model   : {model_path}")
    print(f"  Manifest: {manifest_json_path}")
    print("STOP — Internal ML evaluation complete. External holdout NOT accessed.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 15 ML Model Evaluation")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run in synthetic dry-run mode for pipeline verification",
    )
    args = parser.parse_args()
    run(synthetic=args.synthetic)
