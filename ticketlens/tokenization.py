"""Step 1 - Tokenization study.

Compares naive whitespace splitting with DistilBERT's WordPiece tokenizer,
plots the distribution of WordPiece lengths on the training messages and
picks `max_length` for fine-tuning from the 99th percentile.

Run:  python -m ticketlens.tokenization
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ticketlens import config  # noqa: E402
from ticketlens.data import load_bitext, stratified_split  # noqa: E402
from ticketlens.metrics_io import update_metrics  # noqa: E402

EXAMPLES = [
    "I can't log in to my acc0unt after the password reset!!",
    "where is my refund for order {{Order Number}}?",
    "unsubscribe me from the newsletter pls",
]


def whitespace_tokenize(text: str) -> list[str]:
    """The simplest tokenizer: split on spaces. Punctuation stays glued to words."""
    return text.split()


def choose_max_length(lengths: np.ndarray, percentile: float = 99.0, multiple: int = 8) -> int:
    """Pick max_length = the given percentile, rounded up to a multiple of 8.

    Covering ~99% of messages means almost nothing is truncated while short
    padding keeps training fast. Multiples of 8 are friendlier for GPUs.
    """
    p = float(np.percentile(lengths, percentile))
    return int(np.ceil(p / multiple) * multiple)


def main() -> None:
    from transformers import AutoTokenizer

    config.ensure_dirs()
    tok = AutoTokenizer.from_pretrained(config.DISTILBERT)

    # 1) Side-by-side comparison on a few hand-picked messages.
    comparisons = []
    for text in EXAMPLES:
        ws = whitespace_tokenize(text)
        wp = tok.tokenize(text)  # WordPiece pieces; '##' marks a word continuation
        ids = tok(text)["input_ids"]  # adds [CLS] ... [SEP]
        comparisons.append({"text": text, "whitespace": ws, "wordpiece": wp, "input_ids": ids})
        print(f"\nTEXT       : {text}")
        print(f"whitespace : {ws}  ({len(ws)} tokens)")
        print(f"wordpiece  : {wp}  ({len(wp)} tokens)")
    (config.REPORTS_DIR / "tokenization_examples.json").write_text(json.dumps(comparisons, indent=2), encoding="utf-8")

    # 2) Length distribution on the TRAIN split only (no peeking at test).
    train, _, _ = stratified_split(load_bitext())
    wp_lens = np.array([len(ids) for ids in tok(list(train["text"]))["input_ids"]])
    ws_lens = np.array([len(whitespace_tokenize(t)) for t in train["text"]])
    max_length = choose_max_length(wp_lens)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.arange(0, wp_lens.max() + 2)
    ax.hist(ws_lens, bins=bins, alpha=0.55, label="whitespace words", color="#9aa5b1")
    ax.hist(wp_lens, bins=bins, alpha=0.75, label="WordPiece ids (incl. [CLS]/[SEP])", color="#2a6fdb")
    ax.axvline(max_length, color="#d1495b", linestyle="--", label=f"max_length = {max_length} (p99)")
    ax.set_xlabel("Tokens per message")
    ax.set_ylabel("Messages")
    ax.set_title(f"Token lengths on Bitext train split (n={len(train):,})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / "token_length_hist.png", dpi=110)
    plt.close(fig)

    stats = {
        "n_train_messages": int(len(train)),
        "wordpiece_mean": round(float(wp_lens.mean()), 2),
        "wordpiece_p95": float(np.percentile(wp_lens, 95)),
        "wordpiece_p99": float(np.percentile(wp_lens, 99)),
        "wordpiece_max": int(wp_lens.max()),
        "whitespace_mean": round(float(ws_lens.mean()), 2),
        "chosen_max_length": max_length,
        "truncated_share_at_max_length": round(float((wp_lens > max_length).mean()), 4),
        "vocab_size": int(tok.vocab_size),
    }
    update_metrics("tokenization", "distilbert_wordpiece", stats)
    print("\nLength stats:", json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
