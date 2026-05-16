from __future__ import annotations

import argparse
from contextlib import nullcontext
import math
import time
from pathlib import Path
import sys
from typing import Iterator

import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import GPTConfig, GPTLanguageModel
from src.utils import (
    detect_device,
    ensure_dir,
    format_duration,
    load_model_config,
    load_tokenizer_bundle,
    load_train_config,
    save_json,
    seed_everything,
    setup_logging,
)


def build_grad_scaler(device: str):
    if not torch.cuda.is_available() or device != "cuda":
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda")
    return torch.cuda.amp.GradScaler(enabled=True)


def autocast_context(device: str):
    if device != "cuda":
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return torch.cuda.amp.autocast(enabled=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small GPT-style language model from scratch.")
    parser.add_argument("--model_config", default="configs/tiny_gpt.yaml", help="Model size config YAML path.")
    parser.add_argument("--train_config", default="configs/train_config.yaml", help="Training config YAML path.")
    parser.add_argument("--resume", default=None, help="Optional checkpoint path to resume from.")
    return parser.parse_args()


def create_dataloader(token_file: Path, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    payload = torch.load(token_file, map_location="cpu")
    blocks = payload["blocks"]
    dataset = TensorDataset(blocks)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True if len(dataset) > batch_size else False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def cycle_loader(loader: DataLoader) -> Iterator[torch.Tensor]:
    while True:
        for batch in loader:
            yield batch[0]


@torch.no_grad()
def estimate_loss(model: GPTLanguageModel, loader: DataLoader, device: str, eval_batches: int = 20) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    iterator = iter(loader)
    for _ in range(eval_batches):
        try:
            batch = next(iterator)[0]
        except StopIteration:
            break
        batch = batch.to(device)
        x = batch[:, :-1]
        y = batch[:, 1:]
        _, loss = model(x, y)
        if loss is not None:
            losses.append(float(loss.item()))
    model.train()
    if not losses:
        return float("inf"), float("inf")
    mean_loss = sum(losses) / len(losses)
    perplexity = math.exp(min(mean_loss, 20.0))
    return mean_loss, perplexity


def build_learning_rate(step: int, max_steps: int, base_lr: float, min_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * float(step + 1) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return min_lr + cosine * (base_lr - min_lr)


def save_checkpoint(
    checkpoint_path: Path,
    model: GPTLanguageModel,
    optimizer: torch.optim.Optimizer,
    scaler,
    step: int,
    best_val_loss: float,
    model_cfg: GPTConfig,
) -> None:
    checkpoint = {
        "step": step,
        "best_val_loss": best_val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "model_config": model_cfg.to_dict(),
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)


def main() -> None:
    args = parse_args()
    model_cfg_data = load_model_config(PROJECT_ROOT, args.model_config)
    train_cfg = load_train_config(PROJECT_ROOT, args.train_config)
    seed_everything(train_cfg.seed)

    log_dir = ensure_dir(PROJECT_ROOT / "logs")
    logger = setup_logging("train_gpt", log_file=log_dir / "train.log")
    checkpoints_dir = ensure_dir(PROJECT_ROOT / "checkpoints")

    tokenizer_bundle = load_tokenizer_bundle(PROJECT_ROOT, train_cfg.data.tokenizer_dir)
    actual_vocab_size = len(tokenizer_bundle.special_tokens)
    tokenizer_config_path = PROJECT_ROOT / train_cfg.data.tokenizer_dir / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        import json

        actual_vocab_size = json.loads(tokenizer_config_path.read_text(encoding="utf-8")).get(
            "actual_vocab_size",
            model_cfg_data.vocab_size,
        )

    model_cfg = GPTConfig(
        vocab_size=int(actual_vocab_size),
        block_size=int(model_cfg_data.block_size),
        n_layer=int(model_cfg_data.n_layer),
        n_head=int(model_cfg_data.n_head),
        n_embd=int(model_cfg_data.n_embd),
        dropout=float(model_cfg_data.dropout),
        bias=bool(model_cfg_data.bias),
    )

    device = detect_device()
    logger.info("Training device: %s", device)
    logger.info("Model config: %s", model_cfg.to_dict())

    train_tokens_path = PROJECT_ROOT / train_cfg.data.train_tokens_file
    val_tokens_path = PROJECT_ROOT / train_cfg.data.val_tokens_file
    if not train_tokens_path.exists() or not val_tokens_path.exists():
        raise FileNotFoundError("Tokenized train/val files not found. Run src/tokenize_dataset.py first.")

    train_loader = create_dataloader(train_tokens_path, train_cfg.batch_size, shuffle=True, num_workers=train_cfg.num_workers)
    val_loader = create_dataloader(val_tokens_path, train_cfg.batch_size, shuffle=False, num_workers=train_cfg.num_workers)
    train_iterator = cycle_loader(train_loader)

    model = GPTLanguageModel(model_cfg).to(device)
    logger.info("Model parameters: %.2fM", model.get_num_params() / 1_000_000)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        betas=(0.9, 0.95),
    )
    scaler = build_grad_scaler(device)

    start_step = 0
    best_val_loss = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scaler_state_dict") and device == "cuda":
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_step = int(checkpoint.get("step", 0))
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        logger.info("Resumed training from %s at step %s", args.resume, start_step)

    model.train()
    start_time = time.time()
    running_loss = 0.0

    for step in range(start_step, train_cfg.max_steps):
        optimizer.zero_grad(set_to_none=True)
        step_start = time.time()
        micro_loss_total = 0.0

        for _ in range(train_cfg.gradient_accumulation_steps):
            batch = next(train_iterator).to(device)
            x = batch[:, :-1]
            y = batch[:, 1:]
            with autocast_context(device):
                _, loss = model(x, y)
                if loss is None:
                    raise RuntimeError("Loss should not be None during training.")
                loss = loss / train_cfg.gradient_accumulation_steps
            micro_loss_total += float(loss.item())
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        lr = build_learning_rate(
            step=step,
            max_steps=train_cfg.max_steps,
            base_lr=train_cfg.learning_rate,
            min_lr=train_cfg.min_lr,
            warmup_steps=train_cfg.warmup_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        if scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        step_loss = micro_loss_total
        running_loss += step_loss

        if (step + 1) % 10 == 0 or step == start_step:
            elapsed = time.time() - step_start
            tokens_per_step = (
                train_cfg.batch_size
                * train_cfg.gradient_accumulation_steps
                * model_cfg.block_size
            )
            tokens_per_second = tokens_per_step / max(elapsed, 1e-6)
            logger.info(
                "step=%s/%s loss=%.4f ppl=%.2f lr=%.6f tok/s=%.1f",
                step + 1,
                train_cfg.max_steps,
                step_loss,
                math.exp(min(step_loss, 20.0)),
                lr,
                tokens_per_second,
            )

        if (step + 1) % train_cfg.eval_interval == 0 or step == start_step:
            val_loss, val_ppl = estimate_loss(model, val_loader, device=device)
            logger.info("validation step=%s loss=%.4f ppl=%.2f", step + 1, val_loss, val_ppl)
            if val_loss < best_val_loss:
                best_val_loss = val_loss

        if (step + 1) % train_cfg.save_interval == 0:
            checkpoint_path = checkpoints_dir / f"checkpoint_step_{step + 1}.pt"
            save_checkpoint(checkpoint_path, model, optimizer, scaler, step + 1, best_val_loss, model_cfg)
            save_checkpoint(checkpoints_dir / "last_checkpoint.pt", model, optimizer, scaler, step + 1, best_val_loss, model_cfg)
            logger.info("Saved checkpoint to %s", checkpoint_path)

    final_checkpoint_path = checkpoints_dir / "final_model.pt"
    save_checkpoint(final_checkpoint_path, model, optimizer, scaler, train_cfg.max_steps, best_val_loss, model_cfg)
    total_time = time.time() - start_time
    summary = {
        "final_checkpoint": str(final_checkpoint_path),
        "best_val_loss": best_val_loss,
        "training_time_seconds": round(total_time, 2),
        "training_time_human": format_duration(total_time),
        "model_params": model.get_num_params(),
    }
    save_json(summary, checkpoints_dir / "training_summary.json")
    logger.info("Training finished. Final model saved to %s", final_checkpoint_path)
    print(summary)


if __name__ == "__main__":
    main()
