"""Multiclass metrics, significance tests, and figures."""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: works in CI and on the cluster
import matplotlib.pyplot as plt
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix,
)


def macro_ovr_auc(y_true, prob) -> float:
    """One-vs-rest ROC-AUC averaged over the classes present in y_true (a class
    with no positives has no defined AUC)."""
    aucs = [roc_auc_score(y_true == c, prob[:, c])
            for c in range(prob.shape[1]) if 0 < (y_true == c).sum() < len(y_true)]
    return float(np.mean(aucs)) if aucs else float("nan")


def compute_metrics(y_true, prob) -> dict:
    """prob: (n, n_classes). Macro averages so minority families count equally."""
    n_classes = prob.shape[1]
    y_pred = prob.argmax(1)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=np.unique(y_true), average="macro", zero_division=0)
    per_class = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(n_classes), average=None, zero_division=0)[1]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(p), "recall": float(r), "f1": float(f1),
        "roc_auc": macro_ovr_auc(y_true, prob),
        "per_family_recall": [float(v) for v in per_class],
    }


def mcnemar(y_true, pred_a, pred_b) -> dict:
    """Exact McNemar test on paired predictions over the same samples."""
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    n01, n10 = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return {"a_only_correct": n01, "b_only_correct": n10, "p_value": float(p)}


def evasion_rate(y_true, clean_pred, adv_pred) -> float:
    """Of the samples classified correctly before the attack, the fraction misclassified after."""
    ok = clean_pred == y_true
    return float((adv_pred[ok] != y_true[ok]).mean()) if ok.any() else 0.0


def robustness_auc(budgets, accs) -> float:
    """Area under accuracy-vs-budget, normalised by the budget range (1.0 = no degradation)."""
    b, a = np.asarray(budgets, float), np.asarray(accs, float)
    return float(np.trapezoid(a, b) / (b[-1] - b[0])) if b[-1] > b[0] else float(a[0])


def mean_std(values) -> dict:
    v = np.asarray(values, float)
    return {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v)), "n": int(len(v))}


# ---------------------------------------------------------------- figures
def save_confusion_matrix(y_true, y_pred, families, path, title):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(families))).astype(float)
    cm /= np.maximum(cm.sum(1, keepdims=True), 1)
    size = max(3.2, 0.28 * len(families))
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(families)), families, rotation=90, fontsize=6)
    ax.set_yticks(range(len(families)), families, fontsize=6)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def save_budget_curves(e4: dict, budgets, path):
    """e4[model][attack]["accuracy"] = list of {mean, std} per budget."""
    attacks = sorted({a for m in e4.values() for a in m})
    fig, axes = plt.subplots(1, len(attacks), figsize=(2.6 * len(attacks), 2.6),
                             sharey=True, squeeze=False)
    x = 100 * np.asarray(budgets)
    for ax, atk in zip(axes[0], attacks):
        for model, per_attack in e4.items():
            if atk in per_attack:
                acc = per_attack[atk]["accuracy"]
                ax.errorbar(x, [a["mean"] for a in acc], yerr=[a["std"] for a in acc],
                            marker="o", ms=3, capsize=2, label=model)
        ax.set_title(atk, fontsize=8); ax.set_xlabel("Appended bytes (%)")
    axes[0][0].set_ylabel("Accuracy"); axes[0][0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def save_nn_distance_hist(dists, threshold, path, feature):
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.hist(dists, bins=80, log=True)
    if threshold:
        ax.axvline(threshold, color="red", ls="--", label=f"threshold={threshold}")
        ax.legend(fontsize=7)
    ax.set_xlabel(f"Distance to nearest other sample ({feature})")
    ax.set_ylabel("Samples (log)")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
