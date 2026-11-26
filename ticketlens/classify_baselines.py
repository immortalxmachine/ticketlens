"""Step 2a/2b - Classical baselines for intent classification.

a) TF-IDF (word + bigram counts, re-weighted)  -> Logistic Regression
b) all-MiniLM-L6-v2 sentence embeddings (384-d) -> Logistic Regression

Run:  python -m ticketlens.classify_baselines
"""

import json
import time

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline

from ticketlens import config
from ticketlens.classifier import CATEGORY_MAP_PATH  # noqa: E402
from ticketlens.data import load_bitext, stratified_split
from ticketlens.eval_utils import classification_results, plot_confusion_matrix, top_confusions
from ticketlens.metrics_io import update_metrics


def build_tfidf_model() -> Pipeline:
    """Sparse bag-of-ngrams features + a linear classifier.

    ngram_range=(1, 2) adds bigrams such as "cancel order", which carry intent.
    sublinear_tf dampens words that repeat many times in one message.
    """
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True),
        LogisticRegression(max_iter=2000, C=10.0),
    )


def embed(texts: list[str], model_name: str = config.MINILM, batch_size: int = 256):
    """Turn each message into one dense 384-number vector (mean-pooled MiniLM)."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    # normalize_embeddings=True -> unit length, so dot product == cosine similarity
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)


def evaluate_and_save(name: str, pretty: str, y_test, y_pred, labels, extra: dict) -> dict:
    """Compute metrics, draw the confusion matrix and store everything in metrics.json."""
    result = classification_results(y_test, y_pred, labels)
    result["top_confusions"] = top_confusions(y_test, y_pred)
    result.update(extra)
    plot_confusion_matrix(
        y_test,
        y_pred,
        labels,
        f"{pretty} - test confusion matrix",
        config.REPORTS_DIR / f"confusion_matrix_{name}.png",
    )
    update_metrics("classification", name, result)
    print(f"{pretty:<28} acc={result['accuracy']:.4f}  macro-F1={result['macro_f1']:.4f}")
    return result


def main() -> None:
    config.ensure_dirs()
    train, val, test = stratified_split(load_bitext())
    labels = sorted(train["label"].unique())
    split_info = {"n_train": len(train), "n_val": len(val), "n_test": len(test), "n_classes": len(labels)}
    print("Split sizes:", split_info)
    update_metrics("dataset", "bitext_split", split_info | {"seed": config.SEED})
    # Save the intent -> category map so the API can return the coarse category too.
    categories = dict(sorted(train.groupby("label")["category"].first().items()))
    CATEGORY_MAP_PATH.parent.mkdir(exist_ok=True)
    CATEGORY_MAP_PATH.write_text(json.dumps(categories, indent=2), encoding="utf-8")

    # --- a) TF-IDF + Logistic Regression -----------------------------------
    t0 = time.perf_counter()
    tfidf = build_tfidf_model().fit(train["text"], train["label"])
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = tfidf.predict(test["text"])
    infer_ms = (time.perf_counter() - t0) / len(test) * 1000
    joblib.dump(tfidf, config.MODELS_DIR / "tfidf_lr.joblib")  # served by the API as a fallback
    evaluate_and_save(
        "tfidf_logreg",
        "TF-IDF + LogReg",
        test["label"],
        pred,
        labels,
        {
            "train_seconds": round(train_s, 2),
            "ms_per_message": round(infer_ms, 4),
            "n_features": len(tfidf[0].vocabulary_),
        },
    )

    # --- b) MiniLM sentence embeddings + Logistic Regression ---------------
    t0 = time.perf_counter()
    x_train = embed(list(train["text"]))
    x_test = embed(list(test["text"]))
    embed_s = time.perf_counter() - t0
    clf = LogisticRegression(max_iter=3000, C=10.0).fit(x_train, train["label"])
    pred = clf.predict(x_test)
    evaluate_and_save(
        "minilm_logreg",
        "MiniLM embeddings + LogReg",
        test["label"],
        pred,
        labels,
        {"embed_seconds_train_plus_test": round(embed_s, 2), "embedding_dim": int(x_train.shape[1])},
    )


if __name__ == "__main__":
    main()
