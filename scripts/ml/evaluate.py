"""scripts/ml/evaluate.py
Stage 15 — Metric computation and figure generation for the lncSignal-AD study.

Computes the full metric suite required by the task specification:
  - ROC-AUC, Sensitivity (recall), Specificity, Precision, F1
  - Balanced accuracy, Confusion matrix (TN/FP/FN/TP)
  - 95% bootstrap confidence intervals for AUC
  - Youden-optimal threshold (selected on the same data being evaluated,
    or on a provided training set — never on external/holdout labels)
  - Calibration (Brier score + reliability diagram)
  - Class imbalance reporting

Figures saved:
  - ROC curve (per model and per feature set)
  - Confusion matrix heatmap
  - Calibration reliability diagram
  - Bootstrap AUC CI bar plot

Scientific note:
  All thresholds are selected from the same OOF predictions (Youden criterion
  applied to the OOF ROC curve). This is acceptable because OOF probabilities
  are themselves generated from held-out folds. For strict correctness, the
  threshold should ideally be selected inside the outer CV fold — the current
  approach is a standard practical approximation.

Exports:
  MetricsDict       — TypedDict for the metric return value
  compute_metrics() — full metric computation
  plot_roc_comparison()
  plot_confusion_matrix()
  plot_calibration()
  plot_bootstrap_ci()
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for scripts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

SEED: int = 42
_BOOTSTRAP_N: int = 1000  # bootstrap resamples for CI


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _youden_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Return the probability threshold that maximises Youden's J = Sens + Spec - 1.

    Uses the same OOF predictions being evaluated — this is a practical
    approximation. The returned threshold is NOT fitted on external labels.
    """
    fpr, tpr, thresholds = roc_curve(y_true, proba)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    return float(thresholds[best_idx])


def _bootstrap_auc_ci(
    y_true: np.ndarray,
    proba: np.ndarray,
    n: int = _BOOTSTRAP_N,
    seed: int = SEED,
) -> tuple[float, float]:
    """Stratified bootstrap 95% CI for AUC.

    Stratified by class to ensure both classes are represented in every
    bootstrap sample, which avoids undefined AUC values.
    """
    rng = np.random.default_rng(seed)
    ids0 = np.where(y_true == 0)[0]
    ids1 = np.where(y_true == 1)[0]
    auc_vals: list[float] = []
    for _ in range(n):
        s0 = rng.choice(ids0, size=len(ids0), replace=True)
        s1 = rng.choice(ids1, size=len(ids1), replace=True)
        idx = np.concatenate([s0, s1])
        try:
            auc_vals.append(float(roc_auc_score(y_true[idx], proba[idx])))
        except ValueError:
            continue  # skip degenerate bootstrap samples
    if not auc_vals:
        return (float("nan"), float("nan"))
    lo, hi = float(np.quantile(auc_vals, 0.025)), float(np.quantile(auc_vals, 0.975))
    return lo, hi


def compute_metrics(
    y_true: pd.Series,
    proba: pd.Series,
    threshold: Optional[float] = None,
    label: str = "cv_oof",
) -> dict:
    """Compute the full metric suite for a set of predictions.

    Args:
        y_true: True binary labels (0=Control, 1=AD).
        proba: Predicted probabilities for AD class.
        threshold: Decision threshold. If None, Youden-optimal threshold is
                   computed from the ROC curve of these same predictions.
        label: Evaluation source label — always 'cv_oof' for internal CV;
               'external' only if external holdout (not used in Stage 15).

    Returns:
        dict with keys: auc, auc_ci_lo, auc_ci_hi, sensitivity, specificity,
        precision, recall, f1, balanced_accuracy, brier_score,
        threshold_used, threshold_method, evaluation_source,
        TN, FP, FN, TP, n_ad, n_control, class_imbalance_ratio.
    """
    yt = np.asarray(y_true, dtype=int)
    pb = np.asarray(proba, dtype=float)

    if len(np.unique(yt)) < 2:
        raise ValueError("y_true must contain both classes (0 and 1).")

    auc_val = float(roc_auc_score(yt, pb))
    ci_lo, ci_hi = _bootstrap_auc_ci(yt, pb)

    if threshold is None:
        threshold = _youden_threshold(yt, pb)
        threshold_method = "youden_oof"
    else:
        threshold_method = "provided"

    pred = (pb >= threshold).astype(int)
    cm = confusion_matrix(yt, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")

    n_ad = int(yt.sum())
    n_ctrl = int(len(yt) - n_ad)

    return {
        "evaluation_source": label,
        "auc": auc_val,
        "auc_ci_lo": ci_lo,
        "auc_ci_hi": ci_hi,
        "sensitivity": sens,
        "specificity": spec,
        "precision": float(precision_score(yt, pred, zero_division=0)),
        "recall": float(recall_score(yt, pred, zero_division=0)),
        "f1": float(f1_score(yt, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, pred)),
        "brier_score": float(brier_score_loss(yt, pb)),
        "threshold_used": float(threshold),
        "threshold_method": threshold_method,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "n_ad": n_ad,
        "n_control": n_ctrl,
        "class_imbalance_ratio": round(n_ad / n_ctrl, 4) if n_ctrl > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_roc_comparison(
    results: dict[str, dict],
    y_true: pd.Series,
    oof_probs: dict[str, pd.Series],
    title: str,
    save_path: Path,
) -> None:
    """Plot overlaid OOF ROC curves for multiple models.

    Args:
        results: dict[model_name -> metrics_dict] from compute_metrics().
        y_true: True labels.
        oof_probs: dict[model_name -> oof_probability_series].
        title: Figure title.
        save_path: PNG output path.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    yt = np.asarray(y_true, dtype=int)

    for model_name, proba in oof_probs.items():
        pb = np.asarray(proba, dtype=float)
        fpr, tpr, _ = roc_curve(yt, pb)
        auc_val = roc_auc_score(yt, pb)
        ci_lo = results[model_name].get("auc_ci_lo", float("nan"))
        ci_hi = results[model_name].get("auc_ci_hi", float("nan"))
        label = (
            f"{model_name}  AUC={auc_val:.3f} "
            f"[{ci_lo:.3f}–{ci_hi:.3f}]"
        )
        ax.plot(fpr, tpr, lw=1.8, label=label)

    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Chance (AUC=0.5)")
    ax.set_xlabel("False positive rate (1 − Specificity)", fontsize=12)
    ax.set_ylabel("True positive rate (Sensitivity)", fontsize=12)
    ax.set_title(f"{title}\n(out-of-fold estimates — NOT external validation)", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    print(f"  [evaluate] ROC figure saved: {save_path}")


def plot_confusion_matrix(
    y_true: pd.Series,
    proba: pd.Series,
    threshold: float,
    model_name: str,
    feature_set: str,
    save_path: Path,
) -> None:
    """Save a confusion matrix heatmap.

    Args:
        y_true: True labels (0/1).
        proba: Predicted probabilities.
        threshold: Decision threshold (Youden-optimal from OOF).
        model_name: For figure title.
        feature_set: For figure title.
        save_path: PNG output path.
    """
    yt = np.asarray(y_true, dtype=int)
    pred = (np.asarray(proba, dtype=float) >= threshold).astype(int)
    cm = confusion_matrix(yt, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Control", "AD"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(
        f"{model_name} / {feature_set}\nthreshold={threshold:.3f} (Youden OOF)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_calibration(
    y_true: pd.Series,
    oof_probs: dict[str, pd.Series],
    save_path: Path,
) -> None:
    """Plot reliability diagrams for all models.

    Args:
        y_true: True binary labels.
        oof_probs: dict[model_name -> oof_probability_series].
        save_path: PNG output path.
    """
    yt = np.asarray(y_true, dtype=int)
    n_models = len(oof_probs)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, (model_name, proba) in zip(axes, oof_probs.items()):
        pb = np.asarray(proba, dtype=float)
        prob_true, prob_pred = calibration_curve(yt, pb, n_bins=8, strategy="uniform")
        ax.plot(prob_pred, prob_true, "o-", lw=1.5, label=model_name)
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
        ax.set_xlabel("Mean predicted probability")
        ax.set_title(model_name, fontsize=9)
        brier = brier_score_loss(yt, pb)
        ax.text(0.05, 0.9, f"Brier={brier:.3f}", transform=ax.transAxes, fontsize=8)

    axes[0].set_ylabel("Fraction of positives")
    fig.suptitle(
        "Calibration reliability diagrams (OOF predictions)", fontsize=11, y=1.01
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [evaluate] Calibration figure saved: {save_path}")


def plot_bootstrap_ci(
    summary_df: pd.DataFrame,
    save_path: Path,
) -> None:
    """Bar plot of AUC with 95% bootstrap CI for each model.

    Args:
        summary_df: DataFrame with columns model, feature_set, auc, auc_ci_lo,
                    auc_ci_hi (one row per model × feature_set combination).
        save_path: PNG output path.
    """
    df = summary_df.copy()
    df["label"] = df["model"] + "\n(" + df["feature_set"] + ")"
    df["err_lo"] = df["auc"] - df["auc_ci_lo"]
    df["err_hi"] = df["auc_ci_hi"] - df["auc"]

    fig, ax = plt.subplots(figsize=(max(8, len(df) * 1.2), 5))
    x = np.arange(len(df))
    bars = ax.bar(x, df["auc"], color="steelblue", alpha=0.75, zorder=3)
    ax.errorbar(
        x,
        df["auc"],
        yerr=[df["err_lo"].values, df["err_hi"].values],
        fmt="none",
        color="black",
        capsize=5,
        lw=1.5,
        zorder=4,
    )
    ax.axhline(0.5, color="red", linestyle="--", lw=1.2, label="Chance (AUC=0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], fontsize=8)
    ax.set_ylabel("OOF ROC-AUC")
    ax.set_title(
        "Model comparison — AUC with 95% bootstrap CI\n"
        "(out-of-fold estimates on GSE63061 discovery cohort)",
        fontsize=10,
    )
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [evaluate] Bootstrap CI figure saved: {save_path}")
