"""
data_prep.py — Dataset preparation for QLoRA fine-tuning
Task: Medical Question Answering using MedQuAD dataset
"""

import os
import json
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
MODEL_NAME   = "mistralai/Mistral-7B-Instruct-v0.2"
MAX_LENGTH   = 512
TRAIN_SPLIT  = 0.90
DATASET_NAME = "lavita/ChatDoctor-HealthCareMagic-100k"   # MedQA-style; swap to
# "bigbio/med_qa" or "lavita/medical-qa-datasets" if preferred

OUTPUT_DIR   = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────
# INSTRUCTION TEMPLATE  (Mistral chat format)
# ──────────────────────────────────────────────
def format_example(row: dict) -> dict:
    """
    Wraps a Q/A pair in the Mistral [INST] instruction template.
    Expected columns: 'input' (question) and 'output' (answer).
    Adjust column names if using a different dataset variant.
    """
    question = row.get("input") or row.get("question") or row.get("Question", "")
    answer   = row.get("output") or row.get("answer")  or row.get("Answer",   "")

    text = (
        f"<s>[INST] You are a knowledgeable medical assistant. "
        f"Answer the following medical question accurately and clearly.\n\n"
        f"{question.strip()} [/INST] {answer.strip()} </s>"
    )
    return {"text": text}


# ──────────────────────────────────────────────
# LOAD & FORMAT
# ──────────────────────────────────────────────
def load_and_format(dataset_name: str = DATASET_NAME) -> DatasetDict:
    print(f"[1/4] Loading dataset: {dataset_name}")
    raw = load_dataset(dataset_name, split="train")

    print(f"      Raw size: {len(raw):,} rows")
    # Subsample for feasibility on free-tier GPU (remove cap for full run)
    raw = raw.shuffle(seed=42).select(range(min(20_000, len(raw))))
    print(f"      Working with {len(raw):,} rows after subsampling")

    print("[2/4] Formatting into instruction template …")
    formatted = raw.map(format_example, remove_columns=raw.column_names)

    # Filter out malformed / very short examples
    formatted = formatted.filter(lambda x: len(x["text"]) > 100)

    print(f"      {len(formatted):,} examples after filtering")
    return formatted


# ──────────────────────────────────────────────
# TRAIN / VAL SPLIT
# ──────────────────────────────────────────────
def split_dataset(dataset) -> DatasetDict:
    print("[3/4] Splitting into train / validation …")
    split = dataset.train_test_split(test_size=1 - TRAIN_SPLIT, seed=42)
    splits = DatasetDict({"train": split["train"], "validation": split["test"]})
    print(f"      Train: {len(splits['train']):,}  |  Val: {len(splits['validation']):,}")
    return splits


# ──────────────────────────────────────────────
# TOKENISE  (for length stats — SFTTrainer does its own tokenisation)
# ──────────────────────────────────────────────
def compute_length_stats(dataset, tokenizer) -> dict:
    print("[4/4] Computing token-length statistics …")

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=False)

    tok_ds    = dataset["train"].map(tokenize, batched=True, batch_size=256,
                                     remove_columns=["text"])
    lengths   = [len(ids) for ids in tok_ds["input_ids"]]

    stats = {
        "min":    int(np.min(lengths)),
        "max":    int(np.max(lengths)),
        "mean":   float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "p90":    float(np.percentile(lengths, 90)),
        "p95":    float(np.percentile(lengths, 95)),
        "over_max_length": int(sum(l > MAX_LENGTH for l in lengths)),
    }

    print(f"\n  Token-length stats (train split):")
    for k, v in stats.items():
        print(f"    {k:>16}: {v:.1f}" if isinstance(v, float) else f"    {k:>16}: {v}")

    # Plot
    plt.figure(figsize=(10, 4))
    plt.hist(lengths, bins=60, color="#4F86C6", edgecolor="white", alpha=0.85)
    plt.axvline(MAX_LENGTH, color="#E63946", linewidth=1.5,
                linestyle="--", label=f"max_length={MAX_LENGTH}")
    plt.xlabel("Token length")
    plt.ylabel("Count")
    plt.title("Training Example Token-Length Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/length_distribution.png", dpi=150)
    plt.close()
    print(f"      Histogram saved → {OUTPUT_DIR}/length_distribution.png")

    return stats


# ──────────────────────────────────────────────
# SHOW SAMPLES
# ──────────────────────────────────────────────
def show_samples(dataset, n: int = 3):
    print(f"\n{'='*70}")
    print(f"  {n} SAMPLE TRAINING EXAMPLES")
    print(f"{'='*70}")
    for i, ex in enumerate(dataset["train"].select(range(n))):
        print(f"\n--- Example {i+1} ---")
        print(ex["text"][:600], "…" if len(ex["text"]) > 600 else "")
    print(f"{'='*70}\n")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    formatted = load_and_format()
    splits    = split_dataset(formatted)

    # Save to disk (HuggingFace arrow format)
    splits.save_to_disk(OUTPUT_DIR)
    print(f"\nDataset saved → {OUTPUT_DIR}/")

    # Stats (requires HF token / internet for tokeniser)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    stats = compute_length_stats(splits, tokenizer)
    with open(f"{OUTPUT_DIR}/stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    show_samples(splits)

    print("✅  data_prep.py complete.")
    return splits


if __name__ == "__main__":
    main()
