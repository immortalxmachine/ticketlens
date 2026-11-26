"""Read/write metrics.json.

All numbers in the README come from this file, and this file is only ever
written by code (never edited by hand).
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from ticketlens import config


def load_metrics(path: Path = config.METRICS_PATH) -> dict:
    """Return the current metrics dict (empty if the file does not exist yet)."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def update_metrics(section: str, key: str, value: dict, path: Path = config.METRICS_PATH) -> dict:
    """Store `value` under metrics[section][key] and save the file.

    Merging (instead of overwriting the whole file) lets the classification and
    summarization scripts run independently.
    """
    metrics = load_metrics(path)
    metrics.setdefault(section, {})[key] = value
    metrics["last_updated_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
