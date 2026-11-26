"""API tests with the heavy models replaced by fakes (no downloads)."""

import pytest
from fastapi.testclient import TestClient

from ticketlens import api


class FakeClassifier:
    name = "fake"

    def predict(self, text, k=3):
        return {
            "intent": "cancel_order",
            "category": "ORDER",
            "confidence": 0.98,
            "top_k": [{"intent": "cancel_order", "score": 0.98}][:k],
        }


class FakeSummarizer:
    def summarize(self, text):
        return "Person1 asks to cancel an order."


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api.classifier, "get_classifier", lambda: FakeClassifier())
    monkeypatch.setattr(api.summarizers, "get_summarizer", lambda name: FakeSummarizer())
    monkeypatch.setattr(api, "ollama_up", lambda: False)
    return TestClient(api.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and "bart" in body["summarizers"]


def test_classify(client):
    r = client.post("/classify", json={"text": "cancel my order", "top_k": 1})
    assert r.status_code == 200
    body = r.json()
    assert body["intent"] == "cancel_order" and body["model"] == "fake"
    assert len(body["top_k"]) == 1


def test_classify_rejects_empty_text(client):
    assert client.post("/classify", json={"text": ""}).status_code == 422


def test_summarize_model_selection(client):
    r = client.post("/summarize", json={"text": "#Person1#: cancel it", "model": "flan-t5"})
    assert r.status_code == 200
    assert r.json()["model_id"] == "google/flan-t5-base"


def test_summarize_rejects_unknown_model(client):
    assert client.post("/summarize", json={"text": "hi", "model": "gpt-99"}).status_code == 422


def test_summarize_lead2_real_backend():
    # lead-2 needs no model, so the real summarizer path runs end-to-end.
    text = "#Person1#: a\n#Person2#: b\n#Person1#: c"
    r = TestClient(api.app).post("/summarize", json={"text": text, "model": "lead-2"})
    assert r.json()["summary"] == "#Person1#: a #Person2#: b"
