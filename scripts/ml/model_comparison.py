"""Real GSE63061 internal development study. Run python -m scripts.ml.model_comparison."""
import csv
import gzip
import hashlib
import json
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.ml.data_builder import build_dataset
from scripts.ml.evaluate import (compute_metrics, plot_bootstrap_ci, plot_calibration,
                                plot_confusion_matrix, plot_roc_comparison)
from scripts.ml.models import GROUPS, SEED, build_model_registry
from scripts.ml.nested_cv import checked_splits, run_nested_cv

TABLES = ROOT / "outputs/tables/stage15"
FIGURES = ROOT / "outputs/figures/stage15"
MODELS = ROOT / "outputs/models"
COHORT = "GSE63061"
RULE = ("For the lncRNA-only development model, select family and settings by highest mean "
        "three-fold inner ROC-AUC. Resolve family ties in registry order and grid ties in "
        "ParameterGrid order. Apply the same rule separately on each outer training set "
        "and, after evaluation, on all discovery samples. Fixed AD threshold 0.5.")


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf8")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def discovery():
    data = build_dataset(ROOT / "data", COHORT)
    y = data["y"]
    probes = data["feature_manifest"].feature_id.tolist()
    if probes != data["X_lncrna"].columns.tolist() or not all(data["X_"+g].index.equals(y.index) for g in GROUPS.values()):
        raise ValueError("Stage 14 feature or sample alignment failed")
    if (len(probes), y.value_counts().to_dict()) != (605, {1:139, 0:134}):
        raise ValueError("Unexpected discovery dimensions or encoding")
    # Only the discovery header is read. Array barcodes identify physical chips.
    header = {}
    with gzip.open(ROOT / "data/raw/GSE63061_series_matrix.txt.gz", "rt", encoding="utf8") as stream:
        for row in csv.reader(stream, delimiter="\t"):
            if row and row[0] in {"!Sample_geo_accession","!Sample_title","!Sample_description"}:
                header[row[0]] = row[1:]
            if len(header) == 3:
                break
    info = pd.DataFrame({"title":header["!Sample_title"], "code":header["!Sample_description"]},
                        index=header["!Sample_geo_accession"]).loc[y.index]
    if not info.index.is_unique or not info.code.str.strip().ne("").all() or not info.code.is_unique:
        raise ValueError("Missing or repeated participant proxy; review subject linkage before splitting")
    chip = info.title.str.extract(r"^(\d+)_[A-Za-z]+$", expand=False).rename("chip")
    if chip.isna().any() or not info.title.is_unique:
        raise ValueError("Unresolved chip or array identity")
    return data, chip


def selection_predictions(results):
    """Select by training-only scores independently within each outer fold."""
    reference = next(iter(results.values()))
    fold_id = reference.outer_fold
    scores = pd.Series(np.nan, index=fold_id.index)
    choices = []
    for i in range(reference.n_outer_folds):
        family = max(results, key=lambda name: results[name].fold_best_params[i]["best_inner_cv_auc"])
        held = fold_id.eq(i+1)
        scores.loc[held] = results[family].oof_probabilities.loc[held]
        choices.append(dict(model=family, **results[family].fold_best_params[i]))
    return scores, choices


def freeze(X, y, chip, metrics, manifest):
    registry = build_model_registry(X.shape[1])
    splits = checked_splits(X,y,chip,3,SEED+1)
    searches = {}
    for name,(pipe,grid) in registry.items():
        searches[name] = GridSearchCV(pipe,grid,cv=splits,scoring="roc_auc",refit=False,
                                      error_score="raise").fit(X,y)
    name = max(searches,key=lambda key: searches[key].best_score_)
    params = searches[name].best_params_
    model = clone(registry[name][0]).set_params(**params).fit(X,y)
    transformed = model[:-1].get_feature_names_out().tolist()
    estimator = model.named_steps["model"]
    active = transformed
    if hasattr(estimator,"coef_"):
        active = [p for p,c in zip(transformed,estimator.coef_[0],strict=True) if c != 0]
    file = MODELS/"stage15_development_model.joblib"
    joblib.dump(model,file,compress=3)
    gene = manifest.set_index("feature_id").gene_id
    config = {}
    for key,value in model.get_params(deep=True).items():
        if value is None or isinstance(value,(int,float,str,bool)):
            config[key] = str(value) if isinstance(value,float) and not np.isfinite(value) else value
    frozen = dict(stage="15", discovery_cohort=COHORT, synthetic_dry_run=False, external_holdout_used=False,
                  evaluation_source="cv_oof", best_model_family=name, feature_set="lncrna_only",
                  selection_rule=RULE, threshold=0.5, n_samples=len(y), n_ad=int(y.sum()), n_control=int((y==0).sum()),
                  input_probe_ids=X.columns.tolist(), estimator_input_probe_ids=transformed,
                  active_probe_ids=active, active_gene_ids=gene.loc[active].tolist(),
                  preprocessing=[step for step,_ in model.steps[:-1]], parameters=config,
                  best_params=params, final_cv={k:dict(auc=v.best_score_,params=v.best_params_) for k,v in searches.items()},
                  classes=model.classes_.tolist(), cv_oof_auc=metrics["auc"],
                  cv_oof_auc_ci_95pct=[metrics["auc_ci_lo"],metrics["auc_ci_hi"]],
                  git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                  model_saved_to=file.relative_to(ROOT).as_posix(), model_sha256=digest(file),
                  input_sha256={str(p.relative_to(ROOT).as_posix()):digest(p) for p in
                  [ROOT/"data/processed/stage10/GSE63061_validated_lncrna.parquet",
                   ROOT/"data/interim/GSE63061/metadata.csv",ROOT/"data/annotations/GSE63061_probe_mapping.csv"]},
                  python=platform.python_version(), package_versions={p:version(p) for p in
                  ["scikit-learn","numpy","pandas","scipy","joblib","pyarrow"]},
                  scientific_note="Internal OOF metrics evaluate the lncRNA family-selection procedure. Clinical validity and transfer remain unknown.")
    write_json(MODELS/"stage15_model_manifest.json",frozen)
    return frozen


def run():
    for directory in (TABLES,FIGURES,MODELS):
        directory.mkdir(parents=True,exist_ok=True)
    data,chip = discovery()
    y = data["y"]
    rows = []
    for group,key in GROUPS.items():
        X = data["X_"+key]
        results = run_nested_cv(X,y,build_model_registry(X.shape[1],group),group,groups=chip)
        probabilities = {k:v.oof_probabilities for k,v in results.items()}
        metrics = {}
        for name,result in results.items():
            scores = result.oof_probabilities
            m = compute_metrics(y,scores,groups=chip)
            metrics[name] = m
            rows.append(dict(feature_set=group,model=name,**m))
            pd.DataFrame(dict(true_label=y, oof_probability_ad=scores, outer_fold=result.outer_fold,
                              chip=chip,evaluation_source="cv_oof")).to_csv(TABLES/f"oof_{group}_{name}.csv",index_label="sample_id")
            write_json(TABLES/f"fold_tuning_{group}_{name}.json",result.fold_best_params)
            if group == "lncrna_only":
                plot_confusion_matrix(y,scores,0.5,name,group,FIGURES/f"confusion_{name}_{group}.png")
        plot_roc_comparison(metrics,y,probabilities,f"GSE63061 internal ROC | {group}",FIGURES/f"roc_{group}.png")
        if group == "lncrna_only":
            plot_calibration(y,probabilities,FIGURES/"calibration_lncrna_only.png")
            scores,choices = selection_predictions(results)
            procedure = compute_metrics(y,scores,groups=chip)
            write_json(TABLES/"selection_procedure.json",dict(rule=RULE,folds=choices,metrics=procedure))
            pd.DataFrame(dict(true_label=y,oof_probability_ad=scores,chip=chip,outer_fold=result.outer_fold)
                         ).to_csv(TABLES/"oof_selection_procedure.csv",index_label="sample_id")
    summary = pd.DataFrame(rows)
    summary.to_csv(TABLES/"nested_cv_summary.csv",index=False)
    plot_bootstrap_ci(summary,FIGURES/"bootstrap_auc_ci.png")
    write_json(TABLES/"bootstrap_ci.json",{r["feature_set"]+"__"+r["model"]:
               {k:r[k] for k in ["auc","auc_ci_lo","auc_ci_hi"]} for r in rows})
    frozen = freeze(data["X_lncrna"],y,chip,procedure,data["feature_manifest"])
    print("Frozen model:",frozen["best_model_family"],"Procedure metrics:",procedure,flush=True)


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    with threadpool_limits(limits=1):
        run()
