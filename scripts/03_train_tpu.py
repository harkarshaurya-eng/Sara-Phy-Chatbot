from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure PJRT_DEVICE is set before any torch_xla import — critical for TPU v5e.
os.environ.setdefault("PJRT_DEVICE", "TPU")

from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

from src.formatting import messages_to_text
from src.inference import load_base_model, load_tokenizer_for_model
from src.train_utils import (
    detect_runtime,
    ensure_dir,
    format_duration,
    get_system_prompt,
    get_tpu_memory_info,
    load_config,
    resolve_path,
    save_json,
    seed_everything,
    setup_logging,
    write_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the physics chatbot with LoRA on TPU/GPU/CPU.")
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML config file.")
    parser.add_argument("--resume-from-checkpoint", default=None, help="Optional checkpoint path to resume from.")
    parser.add_argument("--train-path", default=None, help="Optional override for the train split JSONL.")
    parser.add_argument("--eval-path", default=None, help="Optional override for the validation split JSONL.")
    return parser.parse_args()


def build_peft_config(config: dict) -> LoraConfig:
    return LoraConfig(
        r=int(config.get("lora_r", 32)),
        lora_alpha=int(config.get("lora_alpha", 64)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        target_modules=list(config.get("target_modules", [])),
        bias="none",
        task_type="CAUSAL_LM",
    )


def _log_tpu_memory(logger, label: str = "") -> None:
    """Log TPU HBM memory usage if running on TPU."""
    mem = get_tpu_memory_info()
    if mem:
        prefix = f"[{label}] " if label else ""
        logger.info(
            "%sTPU Memory: %.2f GB used / %.2f GB total (%.1f%% utilization)",
            prefix, mem["used_gb"], mem["total_gb"], mem["utilization_pct"],
        )


def _find_latest_checkpoint(output_dir: Path) -> str | None:
    """Find the latest checkpoint directory for auto-resume."""
    checkpoints = sorted(
        [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[-1]) if d.name.split("-")[-1].isdigit() else 0,
    )
    return str(checkpoints[-1]) if checkpoints else None


def main() -> None:
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    log_dir = resolve_path(PROJECT_ROOT, config.get("log_dir", "outputs/logs"))
    logger = setup_logging("train_tpu", log_file=log_dir / "train_tpu.log")

    runtime = detect_runtime()
    seed_everything(int(config.get("seed", 42)))

    logger.info("=" * 60)
    logger.info("PHYSICS CHATBOT TRAINING — START")
    logger.info("=" * 60)
    logger.info("Detected accelerator: %s", runtime.get("accelerator"))
    logger.info("Python: %s | Platform: %s", runtime.get("python"), runtime.get("platform"))

    base_model = str(config.get("base_model"))
    training_mode = str(config.get("training_mode", "lora")).lower()
    if training_mode == "full" and not config.get("enable_full_finetune", False):
        raise ValueError("Full fine-tuning is disabled by default. Set enable_full_finetune=true to allow it.")
    if runtime.get("accelerator") == "tpu" and training_mode == "qlora":
        logger.warning("TPU detected. QLoRA is not supported on TPU, so the run will fall back to standard LoRA.")
        training_mode = "lora"

    train_path = resolve_path(PROJECT_ROOT, args.train_path or config.get("train_split_path", "data/final/train.jsonl"))
    eval_path = resolve_path(PROJECT_ROOT, args.eval_path or config.get("validation_split_path", "data/final/validation.jsonl"))
    if not train_path.exists():
        raise FileNotFoundError(f"Train split not found: {train_path}. Run scripts/02_prepare_dataset.py first.")
    if not eval_path.exists():
        raise FileNotFoundError(f"Validation split not found: {eval_path}. Run scripts/02_prepare_dataset.py first.")

    # Log configuration summary
    logger.info("Base model: %s", base_model)
    logger.info("Training mode: %s", training_mode)
    logger.info("LoRA rank (r): %s, alpha: %s", config.get("lora_r"), config.get("lora_alpha"))
    logger.info("Max seq length: %s", config.get("max_seq_length"))
    logger.info("Batch size: %s, Grad accum: %s → Effective batch: %s",
                config.get("train_batch_size"), config.get("gradient_accumulation_steps"),
                int(config.get("train_batch_size", 1)) * int(config.get("gradient_accumulation_steps", 8)))
    logger.info("Learning rate: %s, Scheduler: %s", config.get("learning_rate"), config.get("lr_scheduler_type", "cosine"))
    logger.info("Epochs: %s, Warmup ratio: %s", config.get("num_train_epochs"), config.get("warmup_ratio"))
    logger.info("Train path: %s", train_path)
    logger.info("Eval path: %s", eval_path)

    _log_tpu_memory(logger, "Before model load")

    tokenizer = load_tokenizer_for_model(base_model, trust_remote_code=bool(config.get("trust_remote_code", True)))
    model, runtime = load_base_model(
        base_model,
        trust_remote_code=bool(config.get("trust_remote_code", True)),
        prefer_bf16=bool(config.get("bf16", True)),
        load_in_4bit=bool(config.get("load_in_4bit", False) and training_mode == "qlora"),
        logger=logger,
    )
    if training_mode == "qlora":
        model = prepare_model_for_kbit_training(model)

    if getattr(model.config, "use_cache", None) is not None:
        model.config.use_cache = False

    _log_tpu_memory(logger, "After model load")

    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Total model parameters: %s (%.2f B)", f"{total_params:,}", total_params / 1e9)

    train_dataset = load_dataset("json", data_files=str(train_path), split="train")
    eval_dataset = load_dataset("json", data_files=str(eval_path), split="train")

    logger.info("Train samples: %s, Eval samples: %s", len(train_dataset), len(eval_dataset))

    def render_example(example: dict) -> dict:
        return {
            "text": messages_to_text(
                example["messages"],
                tokenizer=tokenizer,
                model_name=base_model,
                add_generation_prompt=False,
                enable_thinking=bool(config.get("inference", {}).get("enable_thinking", False)),
            )
        }

    train_dataset = train_dataset.map(render_example, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(render_example, remove_columns=eval_dataset.column_names)

    output_dir = resolve_path(PROJECT_ROOT, config.get("output_dir", "outputs"))
    adapter_dir = ensure_dir(resolve_path(PROJECT_ROOT, config.get("adapter_output_dir", "outputs/adapters")))
    final_adapter_dir = ensure_dir(adapter_dir / "final")
    ensure_dir(output_dir)

    # Determine optimizer — adafactor for TPU (no extra state memory), adamw_torch for GPU
    optimizer_name = "adafactor" if runtime.get("accelerator") == "tpu" else "adamw_torch"
    logger.info("Optimizer: %s", optimizer_name)

    # Determine LR scheduler type from config
    lr_scheduler_type = str(config.get("lr_scheduler_type", "cosine"))

    training_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(config.get("train_batch_size", 1)),
        per_device_eval_batch_size=int(config.get("eval_batch_size", 1)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 8)),
        learning_rate=float(config.get("learning_rate", 1e-4)),
        num_train_epochs=float(config.get("num_train_epochs", 3)),
        warmup_ratio=float(config.get("warmup_ratio", 0.06)),
        weight_decay=float(config.get("weight_decay", 0.01)),
        logging_steps=int(config.get("logging_steps", 5)),
        save_steps=int(config.get("save_steps", 50)),
        eval_steps=int(config.get("eval_steps", 50)),
        save_strategy="steps",
        evaluation_strategy="steps",
        lr_scheduler_type=lr_scheduler_type,
        bf16=bool(config.get("bf16", True) and runtime.get("bf16_supported", False)),
        fp16=False,
        gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
        max_grad_norm=float(config.get("max_grad_norm", 1.0)),
        report_to=["tensorboard"],
        logging_dir=str(log_dir / "tensorboard"),
        remove_unused_columns=False,
        dataset_text_field="text",
        max_seq_length=int(config.get("max_seq_length", 1024)),
        packing=bool(config.get("packing", True)),
        dataloader_num_workers=0,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim=optimizer_name,
    )

    # Build early stopping callback if configured
    callbacks = []
    early_stopping_patience = int(config.get("early_stopping_patience", 0))
    if early_stopping_patience > 0:
        try:
            from transformers import EarlyStoppingCallback
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))
            logger.info("Early stopping enabled with patience=%s", early_stopping_patience)
        except ImportError:
            logger.warning("EarlyStoppingCallback not available in this transformers version, skipping.")

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "dataset_text_field": "text",
        "peft_config": None if training_mode == "full" else build_peft_config(config),
        "callbacks": callbacks if callbacks else None,
    }

    try:
        trainer = SFTTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = SFTTrainer(tokenizer=tokenizer, **trainer_kwargs)

    # Count trainable parameters after LoRA application
    if training_mode != "full":
        trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in trainer.model.parameters() if not p.requires_grad)
        logger.info("Trainable parameters: %s (%.4f%% of total)", f"{trainable:,}", 100.0 * trainable / max(trainable + frozen, 1))

    _log_tpu_memory(logger, "Before training")

    # Auto-resume: check for existing checkpoints if no explicit resume path given
    resume_checkpoint = args.resume_from_checkpoint
    if resume_checkpoint is None:
        latest = _find_latest_checkpoint(output_dir)
        if latest:
            logger.info("Found existing checkpoint, auto-resuming from: %s", latest)
            resume_checkpoint = latest

    logger.info("Starting training...")
    train_start = time.time()

    try:
        train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    except Exception as exc:
        logger.error("Training failed: %s", exc)
        # Try to save what we have before re-raising
        try:
            emergency_dir = ensure_dir(adapter_dir / "emergency_save")
            trainer.save_model(str(emergency_dir))
            tokenizer.save_pretrained(str(emergency_dir))
            logger.info("Emergency checkpoint saved to %s", emergency_dir)
        except Exception:
            logger.error("Could not save emergency checkpoint.")
        raise

    train_elapsed = time.time() - train_start
    logger.info("Training completed in %s", format_duration(train_elapsed))

    _log_tpu_memory(logger, "After training")

    trainer.save_model(str(final_adapter_dir))
    tokenizer.save_pretrained(str(final_adapter_dir))
    metrics = train_result.metrics
    metrics["runtime"] = runtime
    metrics["base_model"] = base_model
    metrics["training_mode"] = training_mode
    metrics["training_duration_seconds"] = round(train_elapsed, 2)
    metrics["training_duration_human"] = format_duration(train_elapsed)
    metrics["lora_r"] = int(config.get("lora_r", 32))
    metrics["lora_alpha"] = int(config.get("lora_alpha", 64))
    metrics["max_seq_length"] = int(config.get("max_seq_length", 1024))
    metrics["effective_batch_size"] = int(config.get("train_batch_size", 1)) * int(config.get("gradient_accumulation_steps", 8))

    save_json(metrics, log_dir / "train_metrics.json")
    write_text(
        "\n".join(
            [
                "# Training Summary",
                "",
                f"- Base model: `{base_model}`",
                f"- Training mode: `{training_mode}`",
                f"- Accelerator: `{runtime.get('accelerator')}`",
                f"- LoRA rank: `{config.get('lora_r')}`, alpha: `{config.get('lora_alpha')}`",
                f"- Max sequence length: `{config.get('max_seq_length')}`",
                f"- Effective batch size: `{metrics['effective_batch_size']}`",
                f"- Learning rate: `{config.get('learning_rate')}` ({lr_scheduler_type})",
                f"- Training duration: `{format_duration(train_elapsed)}`",
                f"- Final adapter directory: `{final_adapter_dir}`",
                f"- Metrics file: `{log_dir / 'train_metrics.json'}`",
                "",
                "```json",
                json.dumps(metrics, indent=2),
                "```",
            ]
        ),
        log_dir / "train_summary.md",
    )

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("Final adapter saved to: %s", final_adapter_dir)
    logger.info("Training duration: %s", format_duration(train_elapsed))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
