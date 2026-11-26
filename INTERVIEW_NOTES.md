# TicketLens: interview notes

Questions an interviewer is likely to ask, each tied to the file that answers it.
Every number quoted here comes from `metrics.json`.

---

### 1. What does the project do, in one minute?
Two NLP tasks a customer-support team needs:
1. **Intent classification**: route a customer message to one of 27 intents (e.g. `cancel_order`, `track_refund`).
   Three models of increasing cost are compared: TF-IDF + logistic regression, frozen MiniLM sentence
   embeddings + logistic regression, and a fine-tuned DistilBERT.
2. **Conversation summarization**: summarize a support dialogue for the next agent. A lead-2 baseline,
   BART, Flan-T5 and a Llama 3.2 instruct model are scored with ROUGE and BERTScore on DialogSum.

Both are served by a FastAPI service (`ticketlens/api.py`) with a Gradio demo (`ticketlens/demo.py`).

### 2. Which datasets, and why?
- **Bitext customer support** (CDLA-Sharing-1.0): 26,872 messages, 27 intents. After dropping exact duplicate
  messages the code keeps 24,635, split 80/10/10 with `stratify=label` (`ticketlens/data.py::stratified_split`).
- **DialogSum** (CC BY-NC-SA 4.0): real-life dialogues with human summaries. A fixed random subset of the test split
  (200 dialogues, seed 42) is used so every model sees the same inputs.
Licenses and caveats are in `data/SOURCES.md`.

### 3. Why drop duplicates before splitting?
Bitext is partly generated from templates, so the same sentence can appear more than once. If one copy lands in train
and another in test, the test score measures memorisation. Deduplicating first (`load_bitext`) removes that leak, and a
test asserts train and test share no text (`tests/test_data_and_metrics.py`).

### 4. Why stratified splitting, and why macro-F1?
Stratifying keeps each intent's share the same in train, val and test, so small intents are not missing from test.
Macro-F1 averages F1 over classes equally, so a model that ignores a rare intent is punished even if accuracy stays
high. `test_classification_results_and_macro_f1` shows exactly that case.

### 5. Explain the tokenization study.
`ticketlens/tokenization.py` compares whitespace splitting with DistilBERT's WordPiece:
- WordPiece splits unknown or misspelled words into known pieces (`acc0unt` -> `acc ##0 ##unt`), so there are no
  out-of-vocabulary tokens; whitespace would treat `reset!!` as one unseen word.
- Messages average 12.89 WordPiece tokens vs 8.87 whitespace tokens on the train split.
- `max_length` is chosen from the train-only length distribution (p99 = 22, rounded up to a multiple of 8 = **24**),
  which truncates only 0.52% of messages. Short sequences make training and inference much cheaper than the 512 default.

### 6. How is DistilBERT fine-tuned?
`ticketlens/classify_distilbert.py` with the Hugging Face `Trainer`:
- `AutoModelForSequenceClassification` adds a randomly initialised classification head on top of the pre-trained encoder.
- 2 epochs, learning rate 5e-5, batch size 16, weight decay 0.01, 6% linear warm-up, fp16 on the GPU.
- `DataCollatorWithPadding` pads each batch only to its longest message (dynamic padding).
- It evaluates on validation every epoch and keeps the best checkpoint by macro-F1 (`load_best_model_at_end`).
- The test set is touched once, at the end. Training took 171.7 s on an RTX 4060 Laptop GPU; log in `reports/train_log.txt`.

### 7. Results, and what do you conclude?
Test set, 2,464 messages:

| Model | Accuracy | Macro-F1 |
|---|---|---|
| TF-IDF + LogReg | 0.9939 | 0.9938 |
| MiniLM embeddings + LogReg | 0.9955 | 0.9952 |
| DistilBERT fine-tuned | 0.9984 | 0.9982 |

The fine-tuned model is best, but the cheap baseline is within 0.5 points. On this dataset TF-IDF is a perfectly
reasonable production choice (1 MB model, CPU, milliseconds). The honest conclusion is that **the data is easy**,
not that DistilBERT is magic. See question 12.

### 8. Why compare against a TF-IDF baseline at all?
Without a baseline you cannot tell whether a transformer is worth its cost. TF-IDF with unigrams and bigrams +
logistic regression trains in seconds on a CPU and sets the bar every heavier model must clear.

### 9. What are frozen embeddings vs fine-tuning?
MiniLM + LogReg encodes every message once with a frozen sentence-transformer and trains only a linear classifier on the
384-d vectors. Fine-tuning updates every encoder weight for the task. Frozen embeddings are cheaper and reusable for
search or clustering; fine-tuning usually wins on accuracy when labelled data exists.

### 10. How do you evaluate summaries?
`ticketlens/eval_summarization.py`:
- **ROUGE-1/2/L** (with stemming): n-gram and longest-common-subsequence overlap with the human reference.
- **BERTScore F1** (roberta-large, layer 17): compares contextual token embeddings, so paraphrases get credit.
- **Latency**: wall-clock seconds per dialogue at batch size 1, after one warm-up call.
- Each model's outputs are saved to `reports/summaries/<model>.jsonl`, and five side-by-side examples go to
  `reports/summarization_examples.md` so the numbers can be checked against real text.

### 11. Why include the lead-2 baseline?
It copies the first two dialogue turns, with no model at all. It shows how much ROUGE a summary gets "for free" from
word overlap; an abstractive model has to beat it to justify its cost. BERTScore values sit in a narrow band
(around 0.85 to 0.92), so differences there are small and should be read together with ROUGE and the examples.

### 12. What are the limitations?
- **Bitext is hybrid-synthetic** (generated from seed templates). Messages are clean and uniform, so all three
  classifiers score above 99%. Real tickets have typos, multiple intents and long threads; expect lower numbers.
- Summarization is zero-shot on 200 dialogues; no model was fine-tuned on DialogSum.
- DialogSum is general everyday dialogue, not support tickets specifically.
- ROUGE rewards wording overlap, not factual correctness. A human review or an LLM judge would be the next step.
- The Ollama Llama 3.2 3B row was **not run** (Ollama was not installed on the build machine); the 1B instruct model
  was run directly with transformers instead (`llama-hf`).
- Docker files are written but **not verified** on this machine.

### 13. How does the API work?
`ticketlens/api.py` (FastAPI, port 8020):
- `POST /classify` -> intent, category, confidence and top-k scores, plus the model name and latency.
- `POST /summarize` -> summary from the chosen model (`lead-2`, `bart`, `flan-t5`, `llama`, `llama-hf`).
- `GET /health` -> status, loaded classifier, available summarizers, whether Ollama is reachable.
Pydantic validates requests (empty text and unknown model names return 422). Models load lazily and are cached
(`functools.cache`), so the first request pays the loading cost once.

### 14. How is it tested without downloading models?
`tests/test_api.py` monkeypatches the classifier and summarizer factories with fakes, so CI needs no torch or
transformers (`requirements-ci.txt`). Real code paths that need no model (TF-IDF, lead-2, ROUGE, metrics merging,
splitting) are tested for real. GitHub Actions runs `ruff check` and `pytest` on every push.

### 15. How do you make results reproducible?
One seed (`config.SEED = 42`) for splits, training and sampling; greedy decoding for the LLMs; `metrics.json` is only
written by code (`ticketlens/metrics_io.py`), and the README tables and chart are generated from it
(`scripts/render_results.py`, `ticketlens/plots.py`).

### 16. What would you do next?
- Test on real (anonymised) tickets, and add an "out of scope" class with a confidence threshold.
- Fine-tune Flan-T5 or a small Llama with LoRA on DialogSum and compare against zero-shot.
- Add an LLM-as-judge or human rating for faithfulness of summaries.
- Quantise the classifier (ONNX / int8) and measure CPU latency.
