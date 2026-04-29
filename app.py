"""
app.py — Gradio demo using HuggingFace Inference API
Deploy on HuggingFace Spaces (CPU free tier — no GPU needed)
"""

import os
import gradio as gr
from huggingface_hub import InferenceClient

HF_TOKEN      = os.environ.get("HF_TOKEN", "")
MODEL_ID      = "mohdomer/mistral-7b-medical-qa-qlora"
SYSTEM_PROMPT = (
    "You are a knowledgeable medical assistant. "
    "Answer the following medical question accurately and clearly.\n\n"
)

client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)

EXAMPLES = [
    "What are the early warning signs of type 2 diabetes?",
    "How does high blood pressure damage the kidneys over time?",
    "What is the difference between viral and bacterial pneumonia?",
    "Can anxiety cause physical chest pain?",
    "What lifestyle changes help manage hypothyroidism symptoms?",
]


def answer(question, max_new_tokens, temperature):
    if not question.strip():
        return "⚠️ Please enter a question."
    prompt = f"<s>[INST] {SYSTEM_PROMPT}{question.strip()} [/INST]"
    try:
        response = client.text_generation(
            prompt,
            max_new_tokens     = int(max_new_tokens),
            temperature        = float(temperature),
            repetition_penalty = 1.1,
            do_sample          = True,
        )
        return response.strip()
    except Exception as e:
        return (
            f"⏳ Model is warming up on HF servers (cold start). "
            f"Please try again in 30 seconds.\n\nError: {str(e)}"
        )


with gr.Blocks(title="Medical QA — Mistral-7B QLoRA") as demo:
    gr.Markdown("""
# 🩺 Medical QA — Mistral-7B Fine-tuned with QLoRA

Fine-tuned **Mistral-7B-Instruct-v0.2** on 20k medical Q&A pairs using **QLoRA**
(4-bit NF4 quantisation + LoRA r=64). Inference via HuggingFace Inference API.

| Metric | Base Model | Fine-tuned | Δ |
|--------|-----------|------------|---|
| ROUGE-1 | 27.00% | 29.45% | +2.45% |
| ROUGE-2 | 3.40% | 5.29% | +1.89% |
| ROUGE-L | 13.16% | 16.37% | +3.21% |
| BLEU | 1.28% | 2.81% | +1.53% |

> ⚠️ Research demo only. Do not use for real medical decisions.
> ⏳ First request may take ~20 seconds (cold start).
""")

    with gr.Row():
        with gr.Column(scale=3):
            question = gr.Textbox(
                label="Your Medical Question",
                placeholder="e.g. What are the symptoms of type 2 diabetes?",
                lines=3,
            )
        with gr.Column(scale=1):
            max_tokens  = gr.Slider(50, 300, value=150, step=50,   label="Max Tokens")
            temperature = gr.Slider(0.1, 1.0, value=0.3, step=0.05, label="Temperature")
            btn = gr.Button("Ask", variant="primary", size="lg")

    output = gr.Textbox(label="Response", lines=10, interactive=False, show_copy_button=True)
    gr.Examples(examples=EXAMPLES, inputs=question, label="Example Questions")

    gr.Markdown("""
---
**Model**: [mohdomer/mistral-7b-medical-qa-qlora](https://huggingface.co/mohdomer/mistral-7b-medical-qa-qlora)  
**Training**: [W&B Run](https://wandb.ai/asratabbssum-lords-institute-of-engineering-and-technology/mistral-medical-qlora/runs/tueb17vn)  
**Code**: [GitHub](https://github.com/MOHD-OMER/llm-finetuning)
""")

    btn.click(fn=answer, inputs=[question, max_tokens, temperature], outputs=output)
    question.submit(fn=answer, inputs=[question, max_tokens, temperature], outputs=output)

demo.launch(ssr_mode=False)
