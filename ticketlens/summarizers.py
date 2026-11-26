"""Summarization back-ends behind one tiny interface: `summarize(text) -> str`.

- lead-2  : extractive baseline, returns the first two dialogue turns (no model)
- bart    : facebook/bart-large-cnn, abstractive, trained on CNN/DailyMail news
- flan-t5 : google/flan-t5-base, instruction-tuned seq2seq model, prompted
- llama   : Llama 3.2 3B served locally by Ollama, prompted via its REST API
- llama-hf: Llama 3.2 1B Instruct run directly with transformers (same prompt;
            used when Ollama is not installed)

Heavy libraries are imported inside the classes so the API and the tests can
import this module without torch installed.
"""

from functools import cache

import requests

from ticketlens import config

# Prompt template for the decoder-only LLM. Kept short and explicit:
# DialogSum references are 1-2 sentences, third person, using #PersonN# tags.
LLAMA_PROMPT = (
    "You summarize customer conversations for a support team.\n"
    "Write a concise summary of the dialogue below in one or two sentences, in the third person. "
    "Refer to the speakers exactly as they appear (e.g. #Person1#, #Person2#). "
    "Output only the summary.\n\n"
    "Dialogue:\n{dialogue}\n\nSummary:"
)
# Public, ungated copy of Meta's Llama 3.2 1B Instruct (Llama 3.2 Community License).
LLAMA_HF_ID = "unsloth/Llama-3.2-1B-Instruct"
FLAN_PROMPT = "Summarize this dialogue:\n{dialogue}"


class Lead2Summarizer:
    """Extractive baseline: copy the first two turns. Shows what ROUGE gives for free."""

    name = "lead-2"

    def summarize(self, text: str) -> str:
        turns = [line.strip() for line in text.splitlines() if line.strip()]
        return " ".join(turns[:2])


class Seq2SeqSummarizer:
    """Encoder-decoder models (BART, T5) run with plain `model.generate`."""

    def __init__(self, name: str, model_id: str, prompt: str | None, max_input_tokens: int):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.name, self.prompt, self.max_input_tokens = name, prompt, max_input_tokens
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(self.device).eval()
        self._torch = torch

    def summarize(self, text: str) -> str:
        source = self.prompt.format(dialogue=text) if self.prompt else text
        # Truncate long inputs to the model's context (BART 1024, T5 512 tokens).
        inputs = self.tokenizer(source, return_tensors="pt", truncation=True, max_length=self.max_input_tokens).to(
            self.device
        )
        with self._torch.no_grad():  # inference only: no gradients, less memory
            out = self.model.generate(
                **inputs,
                num_beams=4,
                max_new_tokens=80,
                min_new_tokens=8,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()


class OllamaSummarizer:
    """Calls a local Ollama server: POST /api/generate with a filled prompt template."""

    name = "llama"

    def __init__(self, host: str = config.OLLAMA_HOST, model: str = config.OLLAMA_MODEL):
        self.host, self.model = host.rstrip("/"), model

    def summarize(self, text: str) -> str:
        resp = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": LLAMA_PROMPT.format(dialogue=text),
                "stream": False,
                # temperature 0 = greedy decoding -> reproducible benchmark numbers
                "options": {"temperature": 0, "num_predict": 100, "seed": config.SEED},
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()


class HFChatSummarizer:
    """Decoder-only chat LLM run with transformers, using the same prompt as Ollama.

    The tokenizer's chat template wraps the prompt in Llama's special
    <|start_header_id|>user ... <|eot_id|> markers before generation.
    """

    def __init__(self, name: str, model_id: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name, self._torch = name, torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # device_map loads weights straight onto the target device (no full extra copy in system RAM).
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map=self.device).eval()

    def summarize(self, text: str) -> str:
        messages = [{"role": "user", "content": LLAMA_PROMPT.format(dialogue=text)}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.device)
        with self._torch.no_grad():
            out = self.model.generate(
                **input_ids,
                max_new_tokens=100,
                do_sample=False,  # greedy = reproducible
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = out[0][input_ids["input_ids"].shape[1] :]  # drop the echoed prompt
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# Name -> factory. The API's `model` field must be one of these keys.
FACTORIES = {
    "lead-2": Lead2Summarizer,
    "bart": lambda: Seq2SeqSummarizer("bart", "facebook/bart-large-cnn", None, 1024),
    "flan-t5": lambda: Seq2SeqSummarizer("flan-t5", "google/flan-t5-base", FLAN_PROMPT, 512),
    "llama": OllamaSummarizer,
    "llama-hf": lambda: HFChatSummarizer("llama-hf", LLAMA_HF_ID),
}
MODEL_IDS = {
    "lead-2": "first two turns (extractive)",
    "bart": "facebook/bart-large-cnn",
    "flan-t5": "google/flan-t5-base",
    "llama": f"{config.OLLAMA_MODEL} via Ollama",
    "llama-hf": f"{LLAMA_HF_ID} via transformers",
}


@cache
def get_summarizer(name: str):
    """Build each summarizer once and reuse it (model loading is the slow part)."""
    if name not in FACTORIES:
        raise KeyError(f"unknown summarizer '{name}', choose from {sorted(FACTORIES)}")
    return FACTORIES[name]()
