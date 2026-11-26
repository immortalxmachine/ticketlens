**Intent classification** - Bitext, 27 intents, stratified split train 19,707 / val 2,464 / test 2,464 (seed 42).

| Model | Accuracy | Macro-F1 | Notes |
|---|---|---|---|
| TF-IDF (1-2 grams) + LogReg | 0.9939 | 0.9938 | 4,977 features, trained in 3.51 s (CPU) |
| all-MiniLM-L6-v2 embeddings + LogReg | 0.9955 | 0.9952 | 384-d embeddings, frozen encoder |
| distilbert-base-uncased fine-tuned | 0.9984 | 0.9982 | 2 epochs, max_length 24, 171.7 s on NVIDIA GeForce RTX 4060 Laptop GPU |

**Dialogue summarization** - DialogSum test subset, n=200 dialogues (seed 42), zero-shot.

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | Avg latency (s) | Avg words | Device |
|---|---|---|---|---|---|---|---|
| lead-2 (`first two turns (extractive)`) | 0.2656 | 0.0573 | 0.2044 | 0.8773 | 0.000 | 25.7 | cpu (no model) |
| bart (`facebook/bart-large-cnn`) | 0.2900 | 0.0867 | 0.2217 | 0.8758 | 1.718 | 30.7 | NVIDIA GeForce RTX 4060 Laptop GPU |
| flan-t5 (`google/flan-t5-base`) | 0.3056 | 0.1051 | 0.2589 | 0.8828 | 1.454 | 13.6 | NVIDIA GeForce RTX 4060 Laptop GPU |
| llama-hf | not run | | | | | | |
| llama | not run | | | | | | |
