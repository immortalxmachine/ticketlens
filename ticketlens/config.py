"""Central settings: paths, random seed and dataset names.

Every script imports from here so a single change (e.g. the seed) applies everywhere.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a local .env file if one exists (see .env.example).
load_dotenv()

# Project root = the folder that contains this package.
ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / os.getenv("MODELS_DIR", "models")
METRICS_PATH = ROOT / "metrics.json"

# One seed for every random operation (splits, training, sampling).
SEED = 42

# Hugging Face dataset ids (both are public, no login needed).
CLS_DATASET = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
SUM_DATASET = "knkarthick/dialogsum"

# Column names in the Bitext dataset.
TEXT_COL = "instruction"  # the customer's message
LABEL_COL = "intent"  # 27 fine-grained intents (we classify these)

# Split ratios for classification: 80% train / 10% val / 10% test.
VAL_FRAC = 0.10
TEST_FRAC = 0.10

# Summarization: how many test dialogues we benchmark on (fixed by SEED).
SUM_TEST_SIZE = int(os.getenv("SUM_TEST_SIZE", "200"))

# Ollama (local LLM server) settings.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# Encoder models.
DISTILBERT = "distilbert-base-uncased"
MINILM = "sentence-transformers/all-MiniLM-L6-v2"


def ensure_dirs() -> None:
    """Create output folders if they are missing."""
    REPORTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
