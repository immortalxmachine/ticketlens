"""Shared evaluation helpers for the three classifiers."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window (works on servers / CI)
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def classification_results(y_true, y_pred, labels: list[str]) -> dict:
    """Compute accuracy, macro-F1 and per-class precision/recall/F1.

    Accuracy  = share of correct predictions (dominated by big classes).
    Macro-F1  = average of per-class F1, every class counts equally,
                so a model that ignores a rare intent is punished.
    """
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    per_class = {
        label: {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f), 4),
            "support": int(s),
        }
        for label, p, r, f, s in zip(labels, precision, recall, f1, support)
    }
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="macro")), 4),
        "n_test": int(len(y_true)),
        "per_class": per_class,
    }


def plot_confusion_matrix(y_true, y_pred, labels: list[str], title: str, out_path: Path) -> None:
    """Save a row-normalised confusion matrix (each row sums to 1 = recall view)."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_xlabel("Predicted intent")
    ax.set_ylabel("True intent")
    ax.set_title(title)
    # Write the off-diagonal mistakes so the confusions are readable.
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j] > 0 and (i != j or cm_norm[i, j] < 0.995):
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=5, color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalised rate")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def top_confusions(y_true, y_pred, k: int = 5) -> list[dict]:
    """Return the k most frequent (true -> predicted) mistakes."""
    pairs: dict[tuple[str, str], int] = {}
    for t, p in zip(y_true, y_pred):
        if t != p:
            pairs[(t, p)] = pairs.get((t, p), 0) + 1
    ranked = sorted(pairs.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [{"true": t, "pred": p, "count": c} for (t, p), c in ranked]
