"""Load a trained intent classifier for serving (API + Gradio).

Two interchangeable back-ends with the same `predict(text, k)` method:
- DistilBertClassifier : the fine-tuned model in models/distilbert-intent
- TfidfClassifier      : the TF-IDF + LogReg pipeline in models/tfidf_lr.joblib
"""

import json
import os
from functools import lru_cache

from ticketlens import config

CATEGORY_MAP_PATH = config.ROOT / "ticketlens" / "assets" / "intent_categories.json"


def load_category_map() -> dict[str, str]:
    """intent -> coarse category (e.g. cancel_order -> ORDER), produced by classify_baselines.py."""
    if CATEGORY_MAP_PATH.exists():
        return json.loads(CATEGORY_MAP_PATH.read_text(encoding="utf-8"))
    return {}


def format_prediction(scores: dict[str, float], k: int, categories: dict[str, str]) -> dict:
    """Turn {label: probability} into the JSON shape returned by the API."""
    top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
    best_label, best_score = top[0]
    return {
        "intent": best_label,
        "category": categories.get(best_label, "UNKNOWN"),
        "confidence": round(float(best_score), 4),
        "top_k": [{"intent": label, "score": round(float(s), 4)} for label, s in top],
    }


class TfidfClassifier:
    name = "tfidf"

    def __init__(self, path=config.MODELS_DIR / "tfidf_lr.joblib"):
        import joblib

        self.pipe = joblib.load(path)
        self.categories = load_category_map()

    def predict(self, text: str, k: int = 3) -> dict:
        probs = self.pipe.predict_proba([text])[0]  # one probability per class
        return format_prediction(dict(zip(self.pipe.classes_, probs)), k, self.categories)


class DistilBertClassifier:
    name = "distilbert"

    def __init__(self, path=config.MODELS_DIR / "distilbert-intent"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device).eval()
        self.categories = load_category_map()

    def predict(self, text: str, k: int = 3) -> dict:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(self.device)
        with self.torch.no_grad():
            logits = self.model(**inputs).logits[0]
        probs = self.torch.softmax(logits, dim=-1).tolist()  # logits -> probabilities
        id2label = self.model.config.id2label
        return format_prediction({id2label[i]: p for i, p in enumerate(probs)}, k, self.categories)


@lru_cache(maxsize=1)
def get_classifier():
    """Return the configured classifier, falling back to TF-IDF if DistilBERT is not trained."""
    wanted = os.getenv("CLASSIFIER", "distilbert")
    if wanted == "distilbert" and (config.MODELS_DIR / "distilbert-intent").exists():
        return DistilBertClassifier()
    return TfidfClassifier()
