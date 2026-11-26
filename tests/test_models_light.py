"""Model code paths that run without torch: TF-IDF baseline, lead-2, ROUGE helper."""

import pytest

from ticketlens.classifier import format_prediction
from ticketlens.classify_baselines import build_tfidf_model
from ticketlens.summarizers import FACTORIES, Lead2Summarizer, get_summarizer


def test_tfidf_baseline_learns_tiny_data(tiny_df):
    model = build_tfidf_model().fit(tiny_df["text"], tiny_df["label"])
    assert model.predict(["please cancel my order"])[0] == "cancel_order"
    assert model.predict(["where is my refund money"])[0] == "track_refund"


def test_format_prediction_orders_scores():
    out = format_prediction({"a": 0.1, "b": 0.7, "c": 0.2}, k=2, categories={"b": "B_CAT"})
    assert out["intent"] == "b" and out["category"] == "B_CAT"
    assert [x["intent"] for x in out["top_k"]] == ["b", "c"]


def test_lead2_is_first_two_turns():
    text = "#Person1#: hi\n#Person2#: hello\n#Person1#: bye"
    assert Lead2Summarizer().summarize(text) == "#Person1#: hi #Person2#: hello"


def test_unknown_summarizer_raises():
    with pytest.raises(KeyError):
        get_summarizer("gpt-99")
    assert {"bart", "flan-t5", "llama", "llama-hf", "lead-2"} <= set(FACTORIES)


def test_rouge_helper():
    pytest.importorskip("rouge_score")
    from ticketlens.eval_summarization import rouge_scores

    s = rouge_scores(["the cat sat"], ["the cat sat"])
    assert s["rouge1"] == 1.0
