"""Model-comparison bar chart drawn straight from metrics.json.

Run:  python -m ticketlens.plots   -> reports/model_comparison.png
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ticketlens import config  # noqa: E402
from ticketlens.metrics_io import load_metrics  # noqa: E402

CLS_NAMES = {
    "tfidf_logreg": "TF-IDF\n+ LogReg",
    "minilm_logreg": "MiniLM emb.\n+ LogReg",
    "distilbert_finetuned": "DistilBERT\nfine-tuned",
}
SUM_ORDER = ["lead-2", "bart", "flan-t5", "llama-hf", "llama"]


def grouped_bars(ax, names, series: dict[str, list[float]], title: str, ylim: tuple[float, float]) -> None:
    """Draw one group of bars per model, one bar per metric, with value labels."""
    x = np.arange(len(names))
    width = 0.8 / len(series)
    colors = ["#2a6fdb", "#f4a259", "#5b8e7d", "#bc4b51"]
    for i, (metric, values) in enumerate(series.items()):
        bars = ax.bar(x + (i - (len(series) - 1) / 2) * width, values, width, label=metric, color=colors[i])
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    ax.set_xticks(x, names, fontsize=8)
    ax.set_ylim(*ylim)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)


def main() -> None:
    m = load_metrics()
    cls = {k: v for k, v in m.get("classification", {}).items() if k in CLS_NAMES}
    summ = {k: m["summarization"][k] for k in SUM_ORDER if k in m.get("summarization", {})}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    n_test = next(iter(cls.values()))["n_test"] if cls else 0
    grouped_bars(
        ax1,
        [CLS_NAMES[k] for k in cls],
        {"accuracy": [v["accuracy"] for v in cls.values()], "macro-F1": [v["macro_f1"] for v in cls.values()]},
        f"Intent classification (Bitext test, n={n_test:,})",
        (0.8, 1.01),
    )
    n_dlg = next(iter(summ.values()))["n_dialogues"] if summ else 0
    grouped_bars(
        ax2,
        list(summ),
        {
            "ROUGE-1": [v["rouge1"] for v in summ.values()],
            "ROUGE-2": [v["rouge2"] for v in summ.values()],
            "ROUGE-L": [v["rougeL"] for v in summ.values()],
            "BERTScore F1": [v["bertscore_f1"] for v in summ.values()],
        },
        f"Dialogue summarization (DialogSum test, n={n_dlg})",
        (0, 1.15),
    )
    ax2.legend(fontsize=8, loc="upper left", ncols=2)  # keep the legend clear of the bars
    fig.tight_layout()
    out = config.REPORTS_DIR / "model_comparison.png"
    fig.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
