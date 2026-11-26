"""Print the README results tables straight from metrics.json (no hand-typed numbers).

Run:  python scripts/render_results.py  > reports/results_tables.md
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
m = json.loads((ROOT / "metrics.json").read_text(encoding="utf-8"))

split = m["dataset"]["bitext_split"]
print(
    f"**Intent classification** - Bitext, {split['n_classes']} intents, stratified split "
    f"train {split['n_train']:,} / val {split['n_val']:,} / test {split['n_test']:,} (seed {split['seed']}).\n"
)
print("| Model | Accuracy | Macro-F1 | Notes |")
print("|---|---|---|---|")
names = {
    "tfidf_logreg": "TF-IDF (1-2 grams) + LogReg",
    "minilm_logreg": "all-MiniLM-L6-v2 embeddings + LogReg",
    "distilbert_finetuned": "distilbert-base-uncased fine-tuned",
}
for key, label in names.items():
    r = m["classification"].get(key)
    if not r:
        print(f"| {label} | not run | not run | |")
        continue
    if key == "tfidf_logreg":
        note = f"{r['n_features']:,} features, trained in {r['train_seconds']} s (CPU)"
    elif key == "minilm_logreg":
        note = f"{r['embedding_dim']}-d embeddings, frozen encoder"
    else:
        note = f"{r['epochs']} epochs, max_length {r['max_length']}, {r['train_seconds']} s on {r['device']}"
    print(f"| {label} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {note} |")

summ = m.get("summarization", {})
if summ:
    n = next(iter(summ.values()))["n_dialogues"]
    print(f"\n**Dialogue summarization** - DialogSum test subset, n={n} dialogues (seed 42), zero-shot.\n")
    print("| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | Avg latency (s) | Avg words | Device |")
    print("|---|---|---|---|---|---|---|---|")
    for key in ["lead-2", "bart", "flan-t5", "llama-hf", "llama"]:
        r = summ.get(key)
        if not r:
            print(f"| {key} | not run | | | | | | |")
            continue
        print(
            f"| {key} (`{r['model_id']}`) | {r['rouge1']:.4f} | {r['rouge2']:.4f} | {r['rougeL']:.4f} | "
            f"{r['bertscore_f1']:.4f} | {r['avg_latency_s']:.3f} | {r['avg_summary_words']} | {r['device']} |"
        )
