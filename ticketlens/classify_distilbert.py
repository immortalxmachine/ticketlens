"""Step 2c - Fine-tune distilbert-base-uncased for intent classification.

A classification head (one linear layer on the [CLS] vector) is added on top of
the pre-trained encoder and the whole network is trained end-to-end with
cross-entropy on our 27 intents.

Run:  python -m ticketlens.classify_distilbert  [--epochs 2]
The terminal output is saved to reports/train_log.txt by the Makefile/README command.
"""

import argparse
import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from ticketlens import config
from ticketlens.classify_baselines import evaluate_and_save
from ticketlens.data import load_bitext, stratified_split
from ticketlens.metrics_io import load_metrics

OUT_DIR = config.MODELS_DIR / "distilbert-intent"


def compute_metrics(eval_pred) -> dict:
    """Called by the Trainer after each epoch on the validation split."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds), "macro_f1": f1_score(labels, preds, average="macro")}


def main(epochs: int = 2, batch_size: int = 32, lr: float = 5e-5) -> None:
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    config.ensure_dirs()
    set_seed(config.SEED)
    use_gpu = torch.cuda.is_available()
    print(f"CUDA available: {use_gpu}" + (f" ({torch.cuda.get_device_name(0)})" if use_gpu else ""))

    train, val, test = stratified_split(load_bitext())
    labels = sorted(train["label"].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    # max_length was chosen from the histogram in tokenization.py (p99 of lengths).
    max_length = load_metrics().get("tokenization", {}).get("distilbert_wordpiece", {}).get("chosen_max_length", 64)
    print(f"Using max_length={max_length}; train={len(train)} val={len(val)} test={len(test)}")

    tok = AutoTokenizer.from_pretrained(config.DISTILBERT)

    def to_dataset(df):
        """pandas -> HF Dataset with input_ids / attention_mask / labels."""
        ds = Dataset.from_dict({"text": list(df["text"]), "label": [label2id[x] for x in df["label"]]})
        # Padding is NOT done here; the collator pads each batch to its longest item.
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=max_length), batched=True)

    train_ds, val_ds, test_ds = to_dataset(train), to_dataset(val), to_dataset(test)

    model = AutoModelForSequenceClassification.from_pretrained(
        config.DISTILBERT, num_labels=len(labels), id2label=id2label, label2id=label2id
    )

    args = TrainingArguments(
        output_dir=str(config.MODELS_DIR / "distilbert-checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=128,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.06,  # small linear warm-up avoids destroying pre-trained weights early
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=50,
        fp16=use_gpu,  # half precision only on GPU
        seed=config.SEED,
        report_to="none",
        disable_tqdm=True,  # plain log lines read better in train_log.txt
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tok),
        processing_class=tok,
        compute_metrics=compute_metrics,
    )

    t0 = time.perf_counter()
    trainer.train()
    train_s = time.perf_counter() - t0

    # Final, untouched test-set evaluation.
    t0 = time.perf_counter()
    logits = trainer.predict(test_ds).predictions
    infer_ms = (time.perf_counter() - t0) / len(test) * 1000
    pred = [id2label[i] for i in np.argmax(logits, axis=-1)]

    trainer.save_model(str(OUT_DIR))
    tok.save_pretrained(str(OUT_DIR))

    evaluate_and_save(
        "distilbert_finetuned",
        "Fine-tuned DistilBERT",
        test["label"],
        pred,
        labels,
        {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "max_length": max_length,
            "train_seconds": round(train_s, 1),
            "ms_per_message_batched": round(infer_ms, 4),
            "device": torch.cuda.get_device_name(0) if use_gpu else "cpu",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    a = parser.parse_args()
    main(a.epochs, a.batch_size, a.lr)
