import os
import yaml
import argparse
from pathlib import Path
from dataclasses import dataclass, field

import gc
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login as hf_login
import wandb
from dotenv import load_dotenv

load_dotenv()  # reads .env file into os.environ


@dataclass
class TrainConfig:
    """All training settings, loaded from configs/train.yaml."""
    model_name: str = "google/gemma-2-2b-it"
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
    fp16: bool = False
    use_4bit: bool = True

    # Data
    train_file: str = "data/train.jsonl"
    val_file: str = "data/val.jsonl"
    max_train_samples: int = None  # None = use full dataset; set to e.g. 800 for a quick run

    # Logging
    logging_steps: int = 25
    save_steps: int = 200
    eval_steps: int = 200
    save_total_limit: int = 3
    wandb_project: str = "tool-use-specialist"

    # Hub
    hub_model_id: str = "sahilmdeshmukh/gemma-2-2b-tool-use-lora"


def load_config(path: str) -> TrainConfig:
    """Load YAML config and merge into TrainConfig dataclass."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return TrainConfig(**{k: v for k, v in data.items() if hasattr(TrainConfig, k)})


def load_model_and_tokenizer(cfg: TrainConfig):
    """Load model with 4-bit NF4 quantization (T4) or fp16 (P100 fallback)."""
    hf_token = os.getenv("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        token=hf_token,
        padding_side="right",
    )

    gc.collect()
    torch.cuda.empty_cache()

    if cfg.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            quantization_config=bnb_config,
            device_map={"": "cuda:0"},
            token=hf_token,
            low_cpu_mem_usage=True,
        )
    else:
        # P100 fallback: fp16, no bitsandbytes (P100 is sm_60, bitsandbytes needs sm_70+)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.float16,
            device_map={"": "cuda:0"},
            token=hf_token,
            low_cpu_mem_usage=True,
        )

    return model, tokenizer


def attach_lora(model, cfg: TrainConfig):
    """Wrap the model with trainable LoRA adapter layers."""

    lora_config = LoraConfig(
        r=cfg.lora_r,                          # Rank: size of the adapter matrices
        lora_alpha=cfg.lora_alpha,             # Scaling factor
        lora_dropout=cfg.lora_dropout,         # Dropout for regularization
        target_modules=cfg.lora_target_modules,
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

    # Filter out examples over 4000 chars (~1024 tokens) — long sequences make steps slow
    train_ds = train_ds.filter(lambda x: len(x["text"]) <= 4000)
    val_ds   = val_ds.filter(lambda x: len(x["text"]) <= 4000)
    print(f"After length filter — train: {len(train_ds):,}  val: {len(val_ds):,}")

    if cfg.max_train_samples is not None:
        train_ds = train_ds.select(range(min(cfg.max_train_samples, len(train_ds))))
        val_ds   = val_ds.select(range(min(cfg.max_train_samples // 8, len(val_ds))))
        print(f"After sample cap    — train: {len(train_ds):,}  val: {len(val_ds):,}")

    return train_ds, val_ds


def train(cfg: TrainConfig):
    """Full training run: load → attach LoRA → train → push to Hub."""

    # --- Auth ---
    hf_login(token=os.getenv("HF_TOKEN"), add_to_git_credential=False)
    wandb.init(project=cfg.wandb_project, config=vars(cfg))

    # --- Build model ---
    model, tokenizer = load_model_and_tokenizer(cfg)
    tokenizer.model_max_length = 1024  # cap sequence length — keeps steps fast
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
        bf16=cfg.bf16 and not cfg.fp16,
        fp16=cfg.fp16,
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
        processing_class=tokenizer,
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
    parser = argparse.ArgumentParser(description="QLoRA fine-tune for tool use")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)
