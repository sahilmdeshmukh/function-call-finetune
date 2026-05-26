import os
import yaml
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import gc
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login as hf_login
import wandb
from dotenv import load_dotenv

load_dotenv()  # reads .env file into os.environ


@dataclass
class TrainConfig:
    """All training settings, loaded from configs/train.yaml."""
    model_name: str = "google/gemma-4-e4b-it"
    output_dir: str = "checkpoints/gemma-tool"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Training
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 1
    max_seq_length: int = 2048
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    bf16: bool = True

    # Data
    train_file: str = "data/train.jsonl"
    val_file: str = "data/val.jsonl"

    # Logging
    logging_steps: int = 25
    save_steps: int = 200
    eval_steps: int = 200
    save_total_limit: int = 3
    wandb_project: str = "tool-use-specialist"

    # Hub
    hub_model_id: str = "sahilmdeshmukh/gemma-4-e4b-tool-use-lora"


def load_config(path: str) -> TrainConfig:
    """Load YAML config and merge into TrainConfig dataclass."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return TrainConfig(**{k: v for k, v in data.items() if hasattr(TrainConfig, k)})


def load_model_and_tokenizer(cfg: TrainConfig):
    """Load Gemma 4 in 4-bit NF4 quantization + its tokenizer."""
    hf_token = os.getenv("HF_TOKEN")

    # --- Tokenizer ---
    # The tokenizer converts raw text into token IDs (numbers) the model understands.
    # padding_side="right" is required by SFTTrainer to avoid shape mismatches.
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        token=hf_token,
        padding_side="right",
    )

    # --- 4-bit quantization config ---
    # This tells bitsandbytes HOW to compress the model weights when loading.
    # NF4 = NormalFloat4, a 4-bit format designed for neural net weights.
    # double_quant = quantize the quantization constants too (saves ~0.4GB extra).
    # compute_dtype = even though weights are stored in 4-bit, actual
    #   matrix multiplications happen in bfloat16 for speed and stability.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # --- Model ---
    # device_map="auto" lets HuggingFace decide which GPU/CPU layers go where.
    # On a single T4, everything lands on cuda:0.
    # Free any leftover GPU memory from previous runs before loading
    gc.collect()
    torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
        low_cpu_mem_usage=True,  # stream weights to GPU instead of loading all in CPU RAM first
    )

    # Manually enable gradient checkpointing instead of prepare_model_for_kbit_training.
    # The latter casts layer norms to float32 which causes a 10GB spike on T4.
    # These two calls give the same training behaviour without touching weight dtypes.
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    return model, tokenizer


def attach_lora(model, cfg: TrainConfig):
    """Wrap the model with trainable LoRA adapter layers."""

    lora_config = LoraConfig(
        r=cfg.lora_r,                          # Rank: size of the adapter matrices
        lora_alpha=cfg.lora_alpha,             # Scaling factor
        lora_dropout=cfg.lora_dropout,         # Dropout for regularization
        # Regex targets only language_model layers — skips vision_tower (Gemma4ClippableLinear)
        # Also includes per_layer_input_gate and per_layer_projection (Gemma 4-specific)
        target_modules=r".*language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|per_layer_input_gate|per_layer_projection)",
        bias="none",                           # Don't train bias terms (saves memory)
        task_type="CAUSAL_LM",                 # We're doing causal language modelling
    )

    model = get_peft_model(model, lora_config)

    # Print how many parameters are actually being trained.
    # You'll see something like: trainable: 20M | total: 4B | 0.5%
    model.print_trainable_parameters()

    return model


def load_datasets(cfg: TrainConfig):
    """Load the JSONL splits produced by data/prepare.py."""

    # load_dataset can read local JSONL files directly.
    # Each line in the file becomes one row in the dataset.
    train_ds = load_dataset("json", data_files=cfg.train_file, split="train")
    val_ds = load_dataset("json", data_files=cfg.val_file, split="train")

    print(f"Train examples : {len(train_ds):,}")
    print(f"Val examples   : {len(val_ds):,}")

    # SFTTrainer expects a column called "text" that contains the full
    # formatted training string. Our prepare.py already saved it that way.
    sample = train_ds[0]["text"]
    print("\n--- Sample training text (first 300 chars) ---")
    print(sample[:300])
    print("----------------------------------------------\n")

    return train_ds, val_ds


def train(cfg: TrainConfig):
    """Full training run: load → attach LoRA → train → push to Hub."""

    # --- Auth ---
    hf_login(token=os.getenv("HF_TOKEN"), add_to_git_credential=False)
    wandb.init(project=cfg.wandb_project, config=vars(cfg))

    # --- Build model ---
    model, tokenizer = load_model_and_tokenizer(cfg)
    model = attach_lora(model, cfg)

    # --- Data ---
    train_ds, val_ds = load_datasets(cfg)

    # --- Training arguments ---
    # SFTConfig is a subclass of HuggingFace's TrainingArguments, extended
    # with SFT-specific options like max_seq_length and dataset_text_field.
    sft_cfg = SFTConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        bf16=cfg.bf16,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        eval_strategy="steps",
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=True,        # At the end, restore the best checkpoint
        metric_for_best_model="eval_loss",  # "best" = lowest validation loss
        report_to="wandb",                  # Send all metrics to Weights & Biases
        dataset_text_field="text",          # Which column in the dataset to train on
        packing=False,                      # Don't pack multiple short examples together
    )

    # --- Trainer ---
    # SFTTrainer handles the training loop, checkpointing, evaluation,
    # and logging for you. You just hand it the model, data, and config.
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    # Resume from checkpoint if one exists (handles Colab disconnects)
    last_checkpoint = None
    checkpoint_dir = Path(cfg.output_dir)
    if checkpoint_dir.exists():
        checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"))
        if checkpoints:
            last_checkpoint = str(checkpoints[-1])
            print(f"Resuming from checkpoint: {last_checkpoint}")

    trainer.train(resume_from_checkpoint=last_checkpoint)

    # --- Save and push ---
    # Save the LoRA adapter locally first, then push to HF Hub.
    # We push the adapter only (NOT the merged model) — standard practice.
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    print(f"\nPushing adapter to: {cfg.hub_model_id}")
    trainer.model.push_to_hub(cfg.hub_model_id, token=os.getenv("HF_TOKEN"))
    tokenizer.push_to_hub(cfg.hub_model_id, token=os.getenv("HF_TOKEN"))

    print("\nDone! Adapter live at:")
    print(f"  https://huggingface.co/{cfg.hub_model_id}")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA fine-tune Gemma 4 E4B")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)
