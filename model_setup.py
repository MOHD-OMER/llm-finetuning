"""
model_setup.py — Load Mistral-7B in 4-bit and apply QLoRA via PEFT
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)


# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

QLORA_CONFIG = dict(
    lora_r          = 64,
    lora_alpha      = 16,
    lora_dropout    = 0.1,
    target_modules  = ["q_proj", "v_proj"],   # key attention projections
    bias            = "none",
    task_type       = TaskType.CAUSAL_LM,
)

BNB_CONFIG = dict(
    load_in_4bit               = True,
    bnb_4bit_quant_type        = "nf4",        # NormalFloat4 — best for LLMs
    bnb_4bit_compute_dtype     = torch.float16,
    bnb_4bit_use_double_quant  = True,         # nested quantisation → saves ~0.4 GB
)


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def print_trainable_params(model):
    """Print trainable vs total parameter count (aim for ~1-2%)."""
    trainable, total = 0, 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    pct = 100 * trainable / total
    print(f"\n  Trainable params : {trainable:>12,}  ({pct:.4f}%)")
    print(f"  Total params     : {total:>12,}")
    print(f"  Frozen params    : {total - trainable:>12,}\n")
    return trainable, total


def get_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved  = torch.cuda.memory_reserved()  / 1e9
        print(f"  GPU memory — allocated: {allocated:.2f} GB | reserved: {reserved:.2f} GB")
    else:
        print("  No GPU detected — running on CPU (training will be very slow)")


# ──────────────────────────────────────────────
# TOKENISER
# ──────────────────────────────────────────────
def load_tokenizer(model_name: str = MODEL_NAME):
    print(f"[1/3] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # Mistral doesn't define a pad token — use EOS
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"   # required for causal LM training
    print(f"      Vocab size: {tokenizer.vocab_size:,}")
    return tokenizer


# ──────────────────────────────────────────────
# BASE MODEL  (4-bit quantised)
# ──────────────────────────────────────────────
def load_base_model(model_name: str = MODEL_NAME):
    print(f"[2/3] Loading base model in 4-bit: {model_name}")
    bnb_config = BitsAndBytesConfig(**BNB_CONFIG)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config  = bnb_config,
        device_map           = "auto",       # spread across available GPUs/CPU
        trust_remote_code    = True,
        torch_dtype          = torch.float16,
    )
    model.config.use_cache           = False   # required for gradient checkpointing
    model.config.pretraining_tp      = 1

    get_gpu_memory()
    return model


# ──────────────────────────────────────────────
# APPLY QLORA
# ──────────────────────────────────────────────
def apply_qlora(model):
    print("[3/3] Applying QLoRA via PEFT …")

    # Cast LayerNorm layers to float32 for stability, freeze base params
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,   # trades compute for memory
    )

    lora_config = LoraConfig(**QLORA_CONFIG)
    model = get_peft_model(model, lora_config)

    print_trainable_params(model)
    get_gpu_memory()
    return model


# ──────────────────────────────────────────────
# MAIN  (returns model + tokenizer for import)
# ──────────────────────────────────────────────
def setup_model_and_tokenizer():
    tokenizer = load_tokenizer()
    model     = load_base_model()
    model     = apply_qlora(model)

    print("✅  model_setup.py complete.")
    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = setup_model_and_tokenizer()
    print(model)
