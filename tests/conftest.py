"""Tiny in-memory fixtures. Nothing here downloads data or models."""

import pandas as pd
import pytest

INTENTS = {
    "cancel_order": [
        "I want to cancel my order",
        "please cancel the order I placed",
        "cancel purchase now",
        "how can I cancel order 123",
        "stop my order please",
        "I need to cancel an order",
        "cancel my recent order",
        "can you cancel the order",
        "help me cancel order",
        "I wish to cancel the purchase",
    ],
    "track_refund": [
        "where is my refund",
        "track my refund please",
        "has my refund been sent",
        "status of refund",
        "when will I get my money back",
        "refund tracking info",
        "check my refund status",
        "I am waiting for a refund",
        "refund not received yet",
        "follow up on my refund",
    ],
    "newsletter_subscription": [
        "unsubscribe me from the newsletter",
        "sign me up for the newsletter",
        "stop newsletter emails",
        "subscribe to newsletter",
        "newsletter subscription help",
        "cancel newsletter",
        "I want your newsletter",
        "remove me from newsletter list",
        "join the mailing newsletter",
        "newsletter sign up",
    ],
}


@pytest.fixture
def tiny_df() -> pd.DataFrame:
    rows = [
        {"text": t, "label": label, "category": label.split("_")[0].upper()}
        for label, texts in INTENTS.items()
        for t in texts
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def tmp_metrics(tmp_path):
    return tmp_path / "metrics.json"
