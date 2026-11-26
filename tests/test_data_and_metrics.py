import numpy as np

from ticketlens.data import stratified_split
from ticketlens.eval_utils import classification_results, plot_confusion_matrix, top_confusions
from ticketlens.metrics_io import load_metrics, update_metrics
from ticketlens.tokenization import choose_max_length, whitespace_tokenize


def test_stratified_split_keeps_every_label_and_no_overlap(tiny_df):
    train, val, test = stratified_split(tiny_df, seed=0)
    assert len(train) + len(val) + len(test) == len(tiny_df)
    assert set(test["label"]) == set(tiny_df["label"])  # stratified: every intent appears in test
    assert not set(train["text"]) & set(test["text"])


def test_split_is_reproducible(tiny_df):
    a = stratified_split(tiny_df, seed=7)[2]["text"].tolist()
    b = stratified_split(tiny_df, seed=7)[2]["text"].tolist()
    assert a == b


def test_classification_results_and_macro_f1():
    y_true = ["a", "a", "a", "b"]
    y_pred = ["a", "a", "a", "a"]  # never predicts the rare class "b"
    res = classification_results(y_true, y_pred, ["a", "b"])
    assert res["accuracy"] == 0.75
    assert res["macro_f1"] < res["accuracy"]  # macro-F1 punishes ignoring "b"
    assert res["per_class"]["b"]["recall"] == 0.0
    assert top_confusions(y_true, y_pred) == [{"true": "b", "pred": "a", "count": 1}]


def test_confusion_matrix_png(tmp_path):
    out = tmp_path / "cm.png"
    plot_confusion_matrix(["a", "b", "b"], ["a", "a", "b"], ["a", "b"], "t", out)
    assert out.stat().st_size > 1000


def test_metrics_merge(tmp_metrics):
    update_metrics("classification", "m1", {"accuracy": 0.9}, path=tmp_metrics)
    update_metrics("summarization", "m2", {"rouge1": 0.4}, path=tmp_metrics)
    m = load_metrics(tmp_metrics)
    assert m["classification"]["m1"]["accuracy"] == 0.9
    assert m["summarization"]["m2"]["rouge1"] == 0.4


def test_tokenization_helpers():
    assert whitespace_tokenize("cancel  my order!") == ["cancel", "my", "order!"]
    assert choose_max_length(np.array([10] * 99 + [50])) % 8 == 0
