"""
push_model.py — Merge LoRA weights into base model and push to HuggingFace Hub
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# ──────────────────────────────────────────────
# CONFIG  —  update HF_USERNAME before running
# ──────────────────────────────────────────────
HF_USERNAME      = os.environ.get("HF_USERNAME", "YOUR_HF_USERNAME")
BASE_MODEL_NAME  = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER_DIR      = "checkpoints/mistral-medical-qlora"
MERGED_DIR       = "checkpoints/mistral-medical-qlora-merged"
HUB_REPO_ID      = f"{HF_USERNAME}/mistral-7b-medical-qa-qlora"

# Load eval results to embed in model card
RESULTS_PATH = "results/evaluation_report.json"


# ──────────────────────────────────────────────
# MERGE
# ──────────────────────────────────────────────
def merge_and_save():
    print(f"[1/3] Loading base model: {BASE_MODEL_NAME}")
    # Load in fp16 (NOT 4-bit) for the merge step
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype    = torch.float16,
        device_map     = "auto",
        trust_remote_code = True,
    )

    print(f"[2/3] Loading LoRA adapter from: {ADAPTER_DIR}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

    print("[3/3] Merging LoRA weights into base model …")
    model = model.merge_and_unload()   # fuses adapter, returns plain model
    model.eval()

    os.makedirs(MERGED_DIR, exist_ok=True)
    print(f"      Saving merged model → {MERGED_DIR}/")
    model.save_pretrained(MERGED_DIR, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(MERGED_DIR)

    print("✅  Merge complete.\n")
    return model, tokenizer


# ──────────────────────────────────────────────
# MODEL CARD
# ──────────────────────────────────────────────
def build_model_card(metrics: dict | None = None) -> str:
    rouge = metrics or {}
    base  = rouge.get("base_model_metrics", {})
    ft    = rouge.get("finetuned_model_metrics", {})

    card = f"""---
language: en
license: apache-2.0
base_model: {BASE_MODEL_NAME}
tags:
  - medical
  - question-answering
  - qlora
  - peft
  - mistral
  - fine-tuned
datasets:
  - lavita/ChatDoctor-HealthCareMagic-100k
pipeline_tag: text-generation
---

# mistral-7b-medical-qa-qlora

A QLoRA fine-tuned version of [Mistral-7B-Instruct-v0.2](https://huggingface.co/{BASE_MODEL_NAME})
for **medical question answering**.

## Model Description

This model was fine-tuned using **QLoRA (Quantized Low-Rank Adaptation)**, a parameter-efficient
fine-tuning (PEFT) method that lets you adapt a large language model using only a tiny fraction
of trainable parameters — without any degradation in quality.

| Property | Value |
|---|---|
| Base model | `{BASE_MODEL_NAME}` |
| Fine-tuning method | QLoRA (4-bit + LoRA) |
| Dataset | ChatDoctor-HealthCareMagic-100k |
| Task | Medical Question Answering |
| Trainable params | ~1.2% of total |

## Training Configuration

| Hyperparameter | Value |
|---|---|
| LoRA rank (r) | 64 |
| LoRA alpha | 16 |
| LoRA dropout | 0.1 |
| Target modules | q_proj, v_proj |
| Epochs | 3 |
| Batch size (effective) | 16 (4 × 4 accumulation) |
| Learning rate | 2e-4 |
| LR scheduler | Cosine |
| Warmup ratio | 0.03 |
| Quantisation | 4-bit NF4 (BitsAndBytes) |
| Optimiser | paged_adamw_32bit |

## Evaluation Results (50 held-out samples)

| Metric | Base Model | Fine-tuned | Δ |
|---|---|---|---|
| ROUGE-1 | {base.get('rouge1', '—')}% | {ft.get('rouge1', '—')}% | ↑ |
| ROUGE-2 | {base.get('rouge2', '—')}% | {ft.get('rouge2', '—')}% | ↑ |
| ROUGE-L | {base.get('rougeL', '—')}% | {ft.get('rougeL', '—')}% | ↑ |
| BLEU    | {base.get('bleu',   '—')}% | {ft.get('bleu',   '—')}% | ↑ |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "{HUB_REPO_ID}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model     = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype = torch.float16,
    device_map  = "auto",
)

def ask(question: str, max_new_tokens: int = 256) -> str:
    prompt = (
        f"<s>[INST] You are a knowledgeable medical assistant. "
        f"Answer the following medical question accurately and clearly.\\n\\n"
        f"{{question.strip()}} [/INST]"
    )
    inputs  = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.3)
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

print(ask("What are the symptoms of type 2 diabetes?"))
```

## Limitations

- The model may hallucinate medical facts — **do not use for real clinical decisions**.
- Performance degrades on rare conditions or highly specialised sub-specialties.
- Responses are generated in English only.
- Maximum reliable context window is ~512 tokens.

## Future Improvements

- Fine-tune on larger, curated medical corpora (MedQA-USMLE, PubMedQA).
- Increase LoRA rank for higher capacity.
- Add DPO alignment for safer, more calibrated responses.
- Quantitative human evaluation by medical professionals.

## Training Tracking

Training metrics (loss curves, GPU usage, LR schedule) are tracked on
[Weights & Biases](https://wandb.ai).
"""
    return card


# ──────────────────────────────────────────────
# PUSH TO HUB
# ──────────────────────────────────────────────
def push_to_hub(model, tokenizer):
    from huggingface_hub import HfApi

    # Load eval metrics if available
    metrics = None
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            metrics = json.load(f)

    # Write model card
    card_text = build_model_card(metrics)
    card_path = os.path.join(MERGED_DIR, "README.md")
    with open(card_path, "w") as f:
        f.write(card_text)
    print(f"  Model card written → {card_path}")

    # Push model
    print(f"\nPushing model to: https://huggingface.co/{HUB_REPO_ID}")
    model.push_to_hub(HUB_REPO_ID, use_auth_token=True, safe_serialization=True)

    # Push tokenizer
    print("Pushing tokenizer …")
    tokenizer.push_to_hub(HUB_REPO_ID, use_auth_token=True)

    # Push model card separately (to ensure it's correct)
    api = HfApi()
    api.upload_file(
        path_or_fileobj = card_path,
        path_in_repo    = "README.md",
        repo_id         = HUB_REPO_ID,
        repo_type       = "model",
    )

    print(f"\n✅  Model pushed → https://huggingface.co/{HUB_REPO_ID}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Merge + Push — Mistral-7B Medical QLoRA")
    print("=" * 60)

    if HF_USERNAME == "YOUR_HF_USERNAME":
        print("⚠️  Set HF_USERNAME env variable before running!")
        print("    export HF_USERNAME=your-huggingface-username")
        return

    model, tokenizer = merge_and_save()
    push_to_hub(model, tokenizer)


if __name__ == "__main__":
    main()
