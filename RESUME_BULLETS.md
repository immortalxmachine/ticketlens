# Resume bullets: TicketLens

All numbers come from `metrics.json` (Bitext test set of 2,464 messages; DialogSum test subset of 200 dialogues) and the pytest suite.

- Fine-tuned DistilBERT for 27-intent customer-support ticket routing, reaching 99.84% accuracy and 0.998 macro-F1 on 2,464 held-out messages in 172 seconds on a laptop GPU.
- Benchmarked TF-IDF, frozen MiniLM embeddings and fine-tuned DistilBERT on a deduplicated, stratified split, showing a 1 MB TF-IDF model comes within 0.5 points of the transformer.
- Compared BART, Flan-T5 and a lead-2 baseline for support-dialogue summarization with ROUGE and BERTScore; Flan-T5-base led with ROUGE-L 0.259 versus 0.222 for BART-large-CNN.
- Served intent classification and summarization through a FastAPI service and Gradio demo, with 17 offline pytest tests and GitHub Actions CI.
