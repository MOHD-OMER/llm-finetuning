"""
train.py — SFTTrainer fine-tuning with full W&B tracking
"""

import os
import json
import wandb
from datasets import load_from_disk
from transformers import TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from model_setup import setup_model_and_tokenizer


# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
DATA_DIR    = "data/processed"
OUTPUT_DIR  = "checkpoints/mistral-medical-qlora"
WANDB_PROJECT = "mistral-medical-qlora"

TRAINING_ARGS = dict(
    output_dir                  = OUTPUT_DIR,
    num_train_epochs            = 3,
    per_device_train_batch_size = 4,
    per_device_eval_batch_size  = 4,
    gradient_accumulation_steps = 4,          # effective batch = 4*4 = 16
    learning_rate               = 2e-4,
    warmup_ratio                = 0.03,
    lr_scheduler_type           = "cosine",
    fp16                        = True,
    logging_steps               = 10,
    eval_steps                  = 100,
    save_steps                  = 200,
    evaluation_strategy         = "steps",
    save_strategy               = "steps",
    load_best_model_at_end      = True,
    metric_for_best_model       = "eval_loss",
    greater_is_better           = False,
    report_to                   = "wandb",
    run_name                    = "mistral-7b-medical-qlora-run1",
    gradient_checkpointing      = True,
    optim                       = "paged_adamw_32bit",  # memory-efficient optimiser
    max_grad_norm               = 0.3,
    group_by_length             = True,        # speeds up training by ~15%
    dataloader_num_workers      = 2,
    ddp_find_unused_parameters  = False,
)

MAX_SEQ_LENGTH = 512


# ──────────────────────────────────────────────
# W&B SETUP
# ──────────────────────────────────────────────
def init_wandb():
    wandb.init(
        project = WANDB_PROJECT,
        config  = TRAINING_ARGS,
        tags    = ["mistral-7b", "qlora", "medical-qa", "peft"],
        notes   = "QLoRA fine-tuning of Mistral-7B-Instruct-v0.2 on medical QA",
    )
    print(f"  W&B run: {wandb.run.url}\n")


# ──────────────────────────────────────────────
# CUSTOM W&B CALLBACK  (GPU memory logging)
# ──────────────────────────────────────────────
from transformers import TrainerCallback
import torch

class WandbGPUCallback(TrainerCallback):
    """Logs GPU memory usage to W&B at each logging step."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if torch.cuda.is_available() and logs is not None:
            logs["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
            logs["gpu_memory_reserved_gb"]  = torch.cuda.memory_reserved()  / 1e9
        return logs


# ──────────────────────────────────────────────
# TRAIN
# ──────────────────────────────────────────────
def train():
    print("=" * 60)
    print("  QLoRA Fine-tuning — Mistral-7B-Instruct-v0.2")
    print("  Task : Medical Question Answering")
    print("=" * 60)

    # 1. Init W&B
    init_wandb()

    # 2. Load model + tokenizer
    model, tokenizer = setup_model_and_tokenizer()

    # 3. Load processed dataset
    print(f"\nLoading dataset from {DATA_DIR} …")
    dataset = load_from_disk(DATA_DIR)
    print(f"  Train: {len(dataset['train']):,}  |  Val: {len(dataset['validation']):,}")

    # 4. Training arguments
    training_args = TrainingArguments(**TRAINING_ARGS)

    # 5. Data collator — only computes loss on completion tokens (answer part)
    #    Response template tells the collator where the answer starts.
    response_template = " [/INST]"
    collator = DataCollatorForCompletionOnlyLM(
        response_template = response_template,
        tokenizer         = tokenizer,
    )

    # 6. SFTTrainer
    trainer = SFTTrainer(
        model             = model,
        args              = training_args,
        train_dataset     = dataset["train"],
        eval_dataset      = dataset["validation"],
        tokenizer         = tokenizer,
        data_collator     = collator,
        dataset_text_field= "text",
        max_seq_length    = MAX_SEQ_LENGTH,
        packing           = False,             # set True for short sequences
        callbacks         = [WandbGPUCallback()],
    )

    # 7. Train!
    print("\n🚀 Starting training …")
    trainer_output = trainer.train()

    # 8. Save adapter weights
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n✅  Adapter saved → {OUTPUT_DIR}/")

    # 9. Log final metrics
    metrics = trainer_output.metrics
    print("\nFinal training metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    with open(f"{OUTPUT_DIR}/training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    wandb.finish()
    return trainer, model, tokenizer


if __name__ == "__main__":
    train()
