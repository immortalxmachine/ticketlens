"""Step 3 - Benchmark summarizers on a fixed DialogSum test subset.

Metrics
- ROUGE-1/2/L : n-gram / longest-common-subsequence overlap with the human summary
- BERTScore F1: similarity of contextual embeddings (config.BERTSCORE_MODEL), so paraphrases count
- latency     : mean wall-clock seconds per dialogue (batch size 1, after warm-up)

Run:  python -m ticketlens.eval_summarization --models lead-2 bart flan-t5 llama
Each model's outputs go to reports/summaries/<model>.jsonl and its scores to metrics.json.
"""

import argparse
import json
import time

import numpy as np

from ticketlens import config
from ticketlens.data import load_dialogsum_test
from ticketlens.metrics_io import update_metrics
from ticketlens.summarizers import MODEL_IDS, get_summarizer

OUT_DIR = config.REPORTS_DIR / "summaries"


def rouge_scores(preds: list[str], refs: list[str]) -> dict:
    """Mean ROUGE F-measures. Stemming makes 'cancelled' match 'cancel'."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    per = [scorer.score(r, p) for p, r in zip(preds, refs)]
    return {k: round(float(np.mean([s[k].fmeasure for s in per])), 4) for k in ["rouge1", "rouge2", "rougeL"]}


def bertscore_f1(preds: list[str], refs: list[str]) -> float:
    """Mean BERTScore F1 using config.BERTSCORE_MODEL (default roberta-large)."""
    import torch
    from bert_score import score
    from bert_score.utils import model2layers

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bert-score's recommended layer per backbone is keyed by the short name (e.g. "roberta-large").
    layers = model2layers.get(config.BERTSCORE_MODEL.split("/")[-1])
    _, _, f1 = score(
        preds, refs, model_type=config.BERTSCORE_MODEL, num_layers=layers, device=device, batch_size=32, verbose=False
    )
    return round(float(f1.mean()), 4)


def free_gpu() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_model(name: str, df) -> tuple[list[str], list[float]]:
    """Summarize every dialogue with one model and time each call."""
    summarizer = get_summarizer(name)
    summarizer.summarize(df["dialogue"][0])  # warm-up: first call pays one-off loading costs
    preds, latencies = [], []
    for i, dialogue in enumerate(df["dialogue"]):
        t0 = time.perf_counter()
        preds.append(summarizer.summarize(dialogue))
        latencies.append(time.perf_counter() - t0)
        if (i + 1) % 25 == 0:
            print(f"  [{name}] {i + 1}/{len(df)}  mean latency {np.mean(latencies):.3f}s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for row_id, pred, lat in zip(df["id"], preds, latencies):
            f.write(json.dumps({"id": row_id, "summary": pred, "latency_s": round(lat, 4)}) + "\n")
    return preds, latencies


def write_examples(df, k: int = 5) -> None:
    """Save k side-by-side examples (reference vs every model that has outputs)."""
    outputs = {}
    for path in sorted(OUT_DIR.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        outputs[path.stem] = {r["id"]: r["summary"] for r in rows}
    lines = ["# Summarization examples (DialogSum test, first 5 of the fixed subset)\n"]
    for _, row in df.head(k).iterrows():
        lines.append(f"## {row['id']}\n\n**Dialogue**\n\n```text\n{row['dialogue']}\n```\n")
        lines.append("| Source | Summary |\n|---|---|")
        lines.append(f"| **Human reference** | {row['summary']} |")
        for name, by_id in outputs.items():
            text = by_id.get(row["id"], "").replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {name} | {text} |")
        lines.append("")
    (config.REPORTS_DIR / "summarization_examples.md").write_text("\n".join(lines), encoding="utf-8")


def main(models: list[str], n: int) -> None:
    import torch

    config.ensure_dirs()
    df = load_dialogsum_test(n)
    refs = list(df["summary"])
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"DialogSum test subset: {len(df)} dialogues (seed {config.SEED}); device: {device}")
    for name in models:
        print(f"\n=== {name} ({MODEL_IDS[name]}) ===")
        preds, latencies = run_model(name, df)
        free_gpu()  # drop the summarizer before loading the BERTScore model (keeps peak memory low)
        result = rouge_scores(preds, refs)
        result["bertscore_f1"] = bertscore_f1(preds, refs)
        result.update(
            {
                "avg_latency_s": round(float(np.mean(latencies)), 4),
                "avg_summary_words": round(float(np.mean([len(p.split()) for p in preds])), 1),
                "n_dialogues": len(df),
                "model_id": MODEL_IDS[name],
                "bertscore_model": config.BERTSCORE_MODEL,
                "device": {"llama": "ollama (local server)", "lead-2": "cpu (no model)"}.get(name, device),
            }
        )
        update_metrics("summarization", name, result)
        print(json.dumps(result, indent=2))
    write_examples(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["lead-2", "bart", "flan-t5", "llama"])
    parser.add_argument("--n", type=int, default=config.SUM_TEST_SIZE)
    a = parser.parse_args()
    main(a.models, a.n)
