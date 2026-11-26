"""Gradio demo UI. It is a thin client: every button calls the FastAPI service.

Run (with the API already on :8020):  python -m ticketlens.demo   -> http://localhost:7870
"""

import os

import gradio as gr
import requests

API_URL = os.getenv("API_URL", "http://localhost:8020").rstrip("/")

CLASSIFY_EXAMPLES = [
    "I was charged twice for the same order, can you help?",
    "how do I change the delivery address for my parcel",
    "I'd like to talk to a real person please",
    "what's your refund policy if the item arrives damaged?",
]
DIALOGUE_EXAMPLE = (
    "#Person1#: Hello, I ordered a laptop last week and it still hasn't arrived.\n"
    "#Person2#: I'm sorry to hear that. Could you give me your order number?\n"
    "#Person1#: Sure, it's 48213.\n"
    "#Person2#: Thanks. It looks like the parcel was held at the warehouse because of a wrong postcode.\n"
    "#Person1#: Oh, I must have mistyped it. Can you fix it?\n"
    "#Person2#: Done. I've corrected the postcode and upgraded you to express shipping, "
    "so it should arrive tomorrow.\n"
    "#Person1#: Great, thank you so much!"
)


def classify(text: str):
    """Call POST /classify and format the answer for the Label + JSON widgets."""
    r = requests.post(f"{API_URL}/classify", json={"text": text, "top_k": 5}, timeout=60)
    r.raise_for_status()
    data = r.json()
    label_scores = {item["intent"]: item["score"] for item in data["top_k"]}
    info = f"**Category:** {data['category']} | **Model:** {data['model']} | {data['latency_ms']} ms"
    return label_scores, info


def summarize(text: str, model: str):
    """Call POST /summarize with the chosen model."""
    r = requests.post(f"{API_URL}/summarize", json={"text": text, "model": model}, timeout=300)
    if r.status_code != 200:
        return f"Error {r.status_code}: {r.json().get('detail')}", ""
    data = r.json()
    return data["summary"], f"**Model:** {data['model_id']} | {data['latency_ms']} ms"


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="TicketLens") as ui:
        gr.Markdown("# TicketLens\nIntent classification and conversation summarization for customer support.")
        with gr.Tab("Classify intent"):
            msg = gr.Textbox(label="Customer message", lines=2, value=CLASSIFY_EXAMPLES[0])
            btn = gr.Button("Classify", variant="primary")
            label = gr.Label(label="Top intents", num_top_classes=5)
            meta = gr.Markdown()
            gr.Examples(CLASSIFY_EXAMPLES, inputs=msg)
            btn.click(classify, inputs=msg, outputs=[label, meta])
        with gr.Tab("Summarize conversation"):
            dialog = gr.Textbox(label="Conversation", lines=9, value=DIALOGUE_EXAMPLE)
            model = gr.Radio(["bart", "flan-t5", "llama-hf", "llama", "lead-2"], value="bart", label="Model")
            sbtn = gr.Button("Summarize", variant="primary")
            summary = gr.Textbox(label="Summary", lines=3)
            smeta = gr.Markdown()
            sbtn.click(summarize, inputs=[dialog, model], outputs=[summary, smeta])
    return ui


if __name__ == "__main__":
    build_ui().launch(server_name=os.getenv("GRADIO_HOST", "127.0.0.1"), server_port=7870)
