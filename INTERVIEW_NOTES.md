# TicketLens: interview notes

20 questions an interviewer is likely to ask, each tied to the file that answers it.
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
not that DistilBERT is magic. See question 16.

**Why a TF-IDF baseline at all?** Without a baseline you cannot tell whether a transformer is worth its cost. TF-IDF with unigrams and bigrams +
logistic regression trains in seconds on a CPU and sets the bar every heavier model must clear.

### 8. What are frozen embeddings vs fine-tuning?
MiniLM + LogReg encodes every message once with a frozen sentence-transformer and trains only a linear classifier on the
384-d vectors. Fine-tuning updates every encoder weight for the task. Frozen embeddings are cheaper and reusable for
search or clustering; fine-tuning usually wins on accuracy when labelled data exists.

### 9. How do you evaluate summaries?
`ticketlens/eval_summarization.py`:
- **ROUGE-1/2/L** (with stemming): n-gram and longest-common-subsequence overlap with the human reference.
- **BERTScore F1** (roberta-large, layer 17): compares contextual token embeddings, so paraphrases get credit.
- **Latency**: wall-clock seconds per dialogue at batch size 1, after one warm-up call.
- Each model's outputs are saved to `reports/summaries/<model>.jsonl`, and five side-by-side examples go to
  `reports/summarization_examples.md` so the numbers can be checked against real text.

### 10. Why include the lead-2 baseline?
It copies the first two dialogue turns, with no model at all. It shows how much ROUGE a summary gets "for free" from
word overlap; an abstractive model has to beat it to justify its cost. On 200 DialogSum test dialogues:

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | Latency (s) |
|---|---|---|---|---|---|
| lead-2 | 0.2656 | 0.0573 | 0.2044 | 0.8773 | 0.000 |
| BART-large-CNN | 0.2900 | 0.0867 | 0.2217 | 0.8758 | 1.718 |
| Flan-T5-base | 0.3056 | 0.1051 | 0.2589 | 0.8828 | 1.454 |

BERTScore sits in a narrow band (0.876 to 0.883), so read it together with ROUGE and the examples.

### 11. Why does BART-large-CNN barely beat copying two lines?
It was fine-tuned on CNN/DailyMail news, where the first sentences of an article are a good summary. It learned
that "lead bias" and, on dialogues, often copies the opening turns almost verbatim (see the first example in
`reports/summarization_examples.md`, where its output equals lead-2). Its BERTScore is even slightly below lead-2.
Flan-T5 was instruction-tuned on many tasks (including dialogue summarization prompts), so it rewrites instead of
copying: shorter outputs (13.6 vs 30.7 words) and the best ROUGE-2/ROUGE-L. Domain match matters more than size
(BART-large has ~400M parameters vs ~250M for Flan-T5-base).

### 12. Is there class imbalance, and how did you handle it?
Mild. After deduplication the 27 intents have between 49 (`cancel_order`) and 100 test messages each (see `per_class`
in `metrics.json`). I handled it with a
**stratified split** (`data.py::stratified_split`) so every split has the same intent proportions, and by reporting
**macro-F1** next to accuracy (`eval_utils.py::classification_results`). No class weights or resampling were needed:
macro-F1 (0.9982) is almost equal to accuracy (0.9984), so the small intents are not being ignored. With strong
imbalance I would add `class_weight="balanced"` to the logistic regressions or a weighted loss in the Trainer.

### 13. Which intents get confused, and does that make sense?
`eval_utils.py::top_confusions` lists the most frequent (true -> predicted) mistakes per model in `metrics.json`.
TF-IDF confuses `contact_customer_service` with `contact_human_agent` and `change_order` with `place_order` (2 each);
MiniLM confuses `get_refund` with `check_refund_policy` (3). These are near-synonyms with overlapping words. DistilBERT
makes only 4 errors on 2,464 messages, all different pairs (e.g. `get_invoice` -> `check_invoice`). The confusion
matrix image is `reports/confusion_matrix_distilbert_finetuned.png`.

### 14. Extractive vs abstractive summarization?
- **Extractive** copies sentences from the input. `summarizers.py::Lead2Summarizer` is the simplest version: the first
  two turns. It is fast and never invents facts, but it keeps the speaker tags and dialogue phrasing, and it misses
  anything decided later in the conversation.
- **Abstractive** models (`Seq2SeqSummarizer` for BART and Flan-T5, `HFChatSummarizer` / `OllamaSummarizer` for Llama)
  generate new text, so they can write "#Person1# wants to know..." in the third person like the human references.
  The risk is hallucination: they can state something the dialogue never said.
- BART-large-CNN behaves almost extractively on this data (question 11), which is why its scores sit close to lead-2.

### 15. Why is lead-2's BERTScore so close to the real models?
BERTScore (`eval_summarization.py::bertscore_f1`) compares contextual token embeddings, and every candidate here is
about the same everyday conversation as the reference: same people, same topic words, same register. RoBERTa
embeddings of on-topic English text are already similar, so scores cluster high (lead-2 0.8773, BART 0.8758, Flan-T5
0.8828) and the useful signal is in small differences. bert-score supports baseline rescaling (`rescale_with_baseline`)
to spread the scale; I kept raw scores so they match the library defaults, and I read them together with ROUGE-2,
where the gap is clearer (0.0573 vs 0.1051).

### 16. What are the limitations?
- **Bitext is hybrid-synthetic** (generated from seed templates). Messages are clean and uniform, so all three
  classifiers score above 99%. Real tickets have typos, multiple intents and long threads; expect lower numbers.
- Summarization is zero-shot on 200 dialogues; no model was fine-tuned on DialogSum.
- DialogSum is general everyday dialogue, not support tickets specifically.
- ROUGE rewards wording overlap, not factual correctness. A human review or an LLM judge would be the next step.
- The Llama rows were **not run**. Ollama was not installed on the build machine, and Llama-3.2-1B-Instruct through
  transformers (`llama-hf`) failed to load because the laptop ran out of Windows paging-file (commit) memory. The
  code paths exist and are selectable in the API; they simply have no measured numbers.
- Docker files are written but **not verified** on this machine.

### 17. How does the API work?
`ticketlens/api.py` (FastAPI, port 8020):
- `POST /classify` -> intent, category, confidence and top-k scores, plus the model name and latency.
- `POST /summarize` -> summary from the chosen model (`lead-2`, `bart`, `flan-t5`, `llama`, `llama-hf`).
- `GET /health` -> status, loaded classifier, available summarizers, whether Ollama is reachable.
Pydantic validates requests (empty text and unknown model names return 422). Models load lazily and are cached
(`functools.cache`), so the first request pays the loading cost once.

### 18. How is it tested without downloading models?
`tests/test_api.py` monkeypatches the classifier and summarizer factories with fakes, so CI needs no torch or
transformers (`requirements-ci.txt`). Real code paths that need no model (TF-IDF, lead-2, ROUGE, metrics merging,
splitting) are tested for real. GitHub Actions runs `ruff check` and `pytest` on every push.

### 19. How do you make results reproducible?
One seed (`config.SEED = 42`) for splits, training and sampling; greedy decoding for the LLMs; `metrics.json` is only
written by code (`ticketlens/metrics_io.py`), and the README tables and chart are generated from it
(`scripts/render_results.py`, `ticketlens/plots.py`).

### 20. What would you do next?
- Test on real (anonymised) tickets, and add an "out of scope" class with a confidence threshold.
- Fine-tune Flan-T5 or a small Llama with LoRA on DialogSum and compare against zero-shot.
- Add an LLM-as-judge or human rating for faithfulness of summaries.
- Quantise the classifier (ONNX / int8) and measure CPU latency.
