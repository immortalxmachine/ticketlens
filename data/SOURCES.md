# Data sources and licenses

Nothing is committed to this repository. Both datasets are downloaded at run time
from the Hugging Face Hub with `datasets.load_dataset` (public, no login needed)
and cached in the local Hugging Face cache. Licenses below were checked on the
dataset cards (Hub API `cardData.license`) on 2026-11-26.

## 1. Intent classification: Bitext customer support

- Hub id: [`bitext/Bitext-customer-support-llm-chatbot-training-dataset`](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
- File: `Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv`
- Size: 26,872 rows, 27 intents in 10+ categories (card says "around 1000 per intent").
  After removing exact duplicate messages the code keeps the count reported in `metrics.json -> dataset.bitext_split`.
- Columns used: `instruction` (customer message), `intent` (label), `category` (coarse label).
- License: **CDLA-Sharing-1.0** (Community Data License Agreement - Sharing, v1.0).
  Use, modification and redistribution are allowed; if the data itself is redistributed
  (we do not), it must stay under the same license.
- Caveat: the card describes the data as *hybrid synthetic* (generated with NLG from
  curated seeds). Messages are cleaner and more uniform than real tickets, which is
  why all models score very high. See INTERVIEW_NOTES.md (limitations).

## 2. Dialogue summarization: DialogSum

- Hub id: [`knkarthick/dialogsum`](https://huggingface.co/datasets/knkarthick/dialogsum)
- Paper: Chen et al., *DialogSum: A Real-Life Scenario Dialogue Summarization Dataset*, Findings of ACL 2021
  (<https://aclanthology.org/2021.findings-acl.449>).
- Size: 13,460 dialogues (+100 holdout); we use a fixed random subset of the **test** split
  (200 dialogues, `seed=42`, see `ticketlens/data.py::load_dialogsum_test`).
- Columns used: `id`, `dialogue`, `summary` (human reference).
- License: **CC BY-NC-SA 4.0** (Attribution - NonCommercial - ShareAlike).
  This project is a non-commercial portfolio / educational benchmark, which the license permits.
  The 5 example dialogues quoted in `reports/summarization_examples.md` are attributed to DialogSum
  and shared under the same license.

## Pre-trained models used (downloaded at run time, not committed)

| Model | Use | License |
|---|---|---|
| distilbert-base-uncased | tokenizer + fine-tuned classifier | Apache-2.0 |
| sentence-transformers/all-MiniLM-L6-v2 | sentence embeddings | Apache-2.0 |
| facebook/bart-large-cnn | summarization | MIT |
| google/flan-t5-base | summarization | Apache-2.0 |
| unsloth/Llama-3.2-1B-Instruct (copy of meta-llama/Llama-3.2-1B-Instruct) | LLM summarization via transformers | Llama 3.2 Community License |
| llama3.2:3b via Ollama (optional) | LLM summarization | Llama 3.2 Community License |
| roberta-large (via bert-score) | BERTScore | MIT |
