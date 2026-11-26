#!/usr/bin/env bash
# Full pipeline, in order. Every number lands in metrics.json.
#   bash scripts/run_all.sh            (Git Bash on Windows, or any Linux/macOS shell)
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
PY=${PY:-.venv/Scripts/python}   # use PY=python on Linux/macOS

$PY -m ticketlens.tokenization                                   # 1. tokens + max_length
$PY -m ticketlens.classify_baselines                             # 2a/2b. TF-IDF and MiniLM
$PY -m ticketlens.classify_distilbert --epochs 2 2>&1 | tee reports/train_log.txt   # 2c. fine-tune
$PY -m ticketlens.eval_summarization --models lead-2 bart flan-t5 llama-hf          # 3. summarizers
# Optional, needs `ollama serve` + `ollama pull llama3.2:3b`:
# $PY -m ticketlens.eval_summarization --models llama
$PY -m ticketlens.plots                                          # comparison chart
