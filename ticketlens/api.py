"""FastAPI service.

Endpoints
- GET  /health     -> liveness + which models are available
- POST /classify   -> intent + category + top-k scores for one message
- POST /summarize  -> summary of a conversation with a selectable model

Run:  uvicorn ticketlens.api:app --port 8020     (Swagger UI at /docs)
"""

import time
from typing import Literal

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ticketlens import classifier, config, summarizers

app = FastAPI(
    title="TicketLens API",
    description="Customer-support intent classification and conversation summarization.",
    version="1.0.0",
)

SummarizerName = Literal["bart", "flan-t5", "llama", "llama-hf", "lead-2"]


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, examples=["I want to cancel my order, it hasn't shipped"])
    top_k: int = Field(3, ge=1, le=10)


class ScoredIntent(BaseModel):
    intent: str
    score: float


class ClassifyResponse(BaseModel):
    intent: str
    category: str
    confidence: float
    top_k: list[ScoredIntent]
    model: str
    latency_ms: float


class SummarizeRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=20000,
        examples=["#Person1#: Hi, my package never arrived.\n#Person2#: Sorry! I'll resend it today."],
    )
    model: SummarizerName = "bart"


class SummarizeResponse(BaseModel):
    summary: str
    model: str
    model_id: str
    latency_ms: float


def ollama_up() -> bool:
    """True if the local Ollama server answers (needed only for model='llama')."""
    try:
        return requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=1.5).ok
    except requests.RequestException:
        return False


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "classifier_models_present": {
            "distilbert": (config.MODELS_DIR / "distilbert-intent").exists(),
            "tfidf": (config.MODELS_DIR / "tfidf_lr.joblib").exists(),
        },
        "summarizers": list(summarizers.FACTORIES),
        "ollama_reachable": ollama_up(),
    }


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest) -> dict:
    t0 = time.perf_counter()
    try:
        clf = classifier.get_classifier()
    except FileNotFoundError as exc:  # no trained model on disk
        raise HTTPException(503, f"classifier not trained yet: {exc}") from exc
    result = clf.predict(req.text, k=req.top_k)
    return result | {"model": clf.name, "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}


@app.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest) -> dict:
    t0 = time.perf_counter()
    try:
        summary = summarizers.get_summarizer(req.model).summarize(req.text)
    except requests.RequestException as exc:  # Ollama not running
        raise HTTPException(503, f"LLM backend unavailable: {exc}") from exc
    return {
        "summary": summary,
        "model": req.model,
        "model_id": summarizers.MODEL_IDS[req.model],
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
    }
