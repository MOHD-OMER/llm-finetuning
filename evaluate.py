"""
evaluate.py — ROUGE + BLEU evaluation with qualitative comparison table
"""

import os
import json
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from evaluate import load as load_metric


# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
BASE_MODEL_NAME  = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER_DIR      = "checkpoints/mistral-medical-qlora"
DATA_DIR         = "data/processed"
RESULTS_DIR      = "results"
NUM_TEST_SAMPLES = 50
MAX_NEW_TOKENS   = 256
TEMPERATURE      = 0.3

os.makedirs(RESULTS_DIR, exist_ok=True)


# ──────────────────────────────────────────────
# MODEL LOADERS
# ──────────────────────────────────────────────
def load_model(model_name: str, adapter_dir: str | None = None):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_quant_type       = "nf4",
        bnb_4bit_compute_dtype    = torch.float16,
        bnb_4bit_use_double_quant = True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config = bnb_config,
        device_map          = "auto",
        torch_dtype         = torch.float16,
    )
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model


def load_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tok.pad_token    = tok.eos_token
    tok.padding_side = "left"   # left-pad for generation
    return tok


# ──────────────────────────────────────────────
# GENERATION
# ──────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a knowledgeable medical assistant. "
    "Answer the following medical question accurately and clearly.\n\n"
)

def build_prompt(question: str) -> str:
    return f"<s>[INST] {SYSTEM_PROMPT}{question.strip()} [/INST]"


@torch.no_grad()
def generate_response(model, tokenizer, question: str) -> str:
    prompt = build_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens  = MAX_NEW_TOKENS,
        temperature     = TEMPERATURE,
        do_sample       = True,
        top_p           = 0.9,
        repetition_penalty = 1.1,
        pad_token_id    = tokenizer.eos_token_id,
    )
    # Strip the prompt from the output
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ──────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────
rouge_metric = load_metric("rouge")
bleu_metric  = load_metric("bleu")


def compute_metrics(predictions: list[str], references: list[str]) -> dict:
    rouge_scores = rouge_metric.compute(
        predictions = predictions,
        references  = references,
        use_stemmer = True,
    )
    # BLEU expects tokenised lists
    bleu_score = bleu_metric.compute(
        predictions = [p.split() for p in predictions],
        references  = [[r.split()] for r in references],
    )
    return {
        "rouge1": round(rouge_scores["rouge1"] * 100, 2),
        "rouge2": round(rouge_scores["rouge2"] * 100, 2),
        "rougeL": round(rouge_scores["rougeL"] * 100, 2),
        "bleu":   round(bleu_score["bleu"]    * 100, 2),
    }


# ──────────────────────────────────────────────
# MAIN EVALUATION LOOP
# ──────────────────────────────────────────────
def evaluate():
    print("=" * 60)
    print("  QLoRA Evaluation — Mistral-7B Medical QA")
    print("=" * 60)

    # Load test samples from validation split
    dataset   = load_from_disk(DATA_DIR)
    val_data  = dataset["validation"].shuffle(seed=99).select(range(NUM_TEST_SAMPLES))

    tokenizer = load_tokenizer(BASE_MODEL_NAME)

    # Parse questions + ground-truth answers from formatted text
    def parse_qa(text: str):
        try:
            q = text.split("[INST]")[1].split("[/INST]")[0].strip()
            # Remove system prompt prefix
            for prefix in [
                "You are a knowledgeable medical assistant. "
                "Answer the following medical question accurately and clearly.\n\n"
            ]:
                q = q.replace(prefix, "").strip()
            a = text.split("[/INST]")[1].replace("</s>", "").strip()
            return q, a
        except Exception:
            return "", ""

    pairs = [parse_qa(ex["text"]) for ex in val_data]
    pairs = [(q, a) for q, a in pairs if q and a]
    print(f"\nEvaluating on {len(pairs)} samples …\n")

    # ── Base model ──
    print("Loading base model (no adapter) …")
    base_model = load_model(BASE_MODEL_NAME)
    base_preds = []
    for q, _ in tqdm(pairs, desc="Base model inference"):
        base_preds.append(generate_response(base_model, tokenizer, q))
    del base_model
    torch.cuda.empty_cache()

    # ── Fine-tuned model ──
    print("\nLoading fine-tuned model (with QLoRA adapter) …")
    ft_model = load_model(BASE_MODEL_NAME, adapter_dir=ADAPTER_DIR)
    ft_preds = []
    for q, _ in tqdm(pairs, desc="Fine-tuned model inference"):
        ft_preds.append(generate_response(ft_model, tokenizer, q))
    del ft_model
    torch.cuda.empty_cache()

    # ── Metrics ──
    ground_truths = [a for _, a in pairs]
    questions     = [q for q, _ in pairs]

    base_metrics = compute_metrics(base_preds, ground_truths)
    ft_metrics   = compute_metrics(ft_preds,   ground_truths)

    print("\n" + "─" * 60)
    print(f"  {'Metric':<12} {'Base Model':>12} {'Fine-tuned':>12} {'Δ':>8}")
    print("─" * 60)
    for k in ["rouge1", "rouge2", "rougeL", "bleu"]:
        delta = ft_metrics[k] - base_metrics[k]
        arrow = "↑" if delta > 0 else "↓"
        print(f"  {k:<12} {base_metrics[k]:>11.2f}% {ft_metrics[k]:>11.2f}% {arrow}{abs(delta):>6.2f}%")
    print("─" * 60)

    # ── Qualitative examples (top-5 where fine-tuned wins most) ──
    improvements = []
    for i, (q, gt, bp, fp) in enumerate(zip(questions, ground_truths, base_preds, ft_preds)):
        base_r = rouge_metric.compute(predictions=[bp], references=[gt], use_stemmer=True)
        ft_r   = rouge_metric.compute(predictions=[fp], references=[gt], use_stemmer=True)
        delta  = ft_r["rougeL"] - base_r["rougeL"]
        improvements.append((delta, i, q, gt, bp, fp))

    improvements.sort(reverse=True)
    top5 = improvements[:5]

    print("\n📋 TOP 5 EXAMPLES — Fine-tuned clearly wins:")
    qualitative = []
    for rank, (delta, idx, q, gt, bp, fp) in enumerate(top5, 1):
        print(f"\n{'='*60}")
        print(f"  Example {rank}  (ROUGE-L improvement: +{delta*100:.1f}%)")
        print(f"  Q: {q[:200]}")
        print(f"\n  Base  : {bp[:300]}")
        print(f"\n  FT    : {fp[:300]}")
        print(f"\n  Truth : {gt[:300]}")
        qualitative.append({
            "rank": rank,
            "question": q,
            "base_response": bp,
            "finetuned_response": fp,
            "ground_truth": gt,
            "rougeL_improvement": round(delta * 100, 2),
        })

    # ── Full comparison table (50 rows) ──
    comparison_table = [
        {
            "question":          q,
            "base_response":     bp,
            "finetuned_response": fp,
            "ground_truth":      gt,
        }
        for q, gt, bp, fp in zip(questions, ground_truths, base_preds, ft_preds)
    ]

    # ── Save report ──
    report = {
        "num_samples":       len(pairs),
        "max_new_tokens":    MAX_NEW_TOKENS,
        "temperature":       TEMPERATURE,
        "base_model_metrics":    base_metrics,
        "finetuned_model_metrics": ft_metrics,
        "qualitative_examples":  qualitative,
        "full_comparison_table": comparison_table,
    }
    report_path = f"{RESULTS_DIR}/evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n✅  Evaluation report saved → {report_path}")

    return report


if __name__ == "__main__":
    evaluate()
