from __future__ import annotations

import json
import logging
import os
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

try:
    import numpy as np
except Exception:  # pragma: no cover - optional at import time
    np = None

try:
    import torch
except Exception:  # pragma: no cover - optional at import time
    torch = None


SYSTEM_PROMPT = (
    "You are PhysicsGPT, a helpful physics tutor. Explain physics clearly and solve problems step by step."
)


@dataclass
class ModelSizeConfig:
    name: str
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float
    bias: bool
    tokenizer_vocab_size: int


@dataclass
class SplitConfig:
    train: float = 0.90
    val: float = 0.05
    test: float = 0.05


@dataclass
class DataPathsConfig:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    tokenizer_dir: str = "data/tokenizer"
    local_text_dir: str = "data/raw/local"
    sample_dir: str = "data/sample"
    max_samples_per_dataset: int = 5000
    corpus_file: str = "data/processed/corpus.txt"
    train_text_file: str = "data/processed/train.txt"
    val_text_file: str = "data/processed/val.txt"
    test_text_file: str = "data/processed/test.txt"
    train_tokens_file: str = "data/processed/train_tokens.pt"
    val_tokens_file: str = "data/processed/val_tokens.pt"
    test_tokens_file: str = "data/processed/test_tokens.pt"
    manifest_file: str = "data/raw/dataset_manifest.json"
    sample_records_file: str = "data/sample/sample_records.jsonl"
    sample_corpus_file: str = "data/sample/sample_corpus.txt"


@dataclass
class PreprocessingConfig:
    min_question_chars: int = 8
    min_answer_chars: int = 20
    min_text_chars: int = 80
    max_text_chars: int = 2000
    stride_ratio: float = 0.5


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    max_steps: int = 5000
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 200
    eval_interval: int = 250
    save_interval: int = 500
    max_grad_norm: float = 1.0
    weight_decay: float = 0.1
    num_workers: int = 0
    train_val_test_split: SplitConfig = field(default_factory=SplitConfig)
    data: DataPathsConfig = field(default_factory=DataPathsConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    tokenizer: dict[str, Any] = field(default_factory=lambda: {"special_tokens": []})
    generation: dict[str, Any] = field(
        default_factory=lambda: {
            "temperature": 0.8,
            "top_k": 50,
            "top_p": 0.95,
            "max_new_tokens": 200,
        }
    )


@dataclass
class TokenizerBundle:
    tokenizer_path: Path
    special_tokens: dict[str, int]


def project_root_from_file(file_path: str | Path) -> Path:
    return Path(file_path).resolve().parents[1]


def resolve_path(project_root: str | Path, value: str | Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return Path(project_root).resolve() / candidate


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def ensure_parent(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    return path_obj.parent


def load_yaml(path: str | Path) -> dict[str, Any]:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Missing YAML file: {path_obj}")
    with path_obj.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_model_config(project_root: str | Path, config_path: str | Path) -> ModelSizeConfig:
    payload = load_yaml(resolve_path(project_root, config_path))
    return ModelSizeConfig(
        name=str(payload.get("name", "tiny_gpt")),
        vocab_size=int(payload.get("vocab_size", 16000)),
        block_size=int(payload.get("block_size", 256)),
        n_layer=int(payload.get("n_layer", 4)),
        n_head=int(payload.get("n_head", 4)),
        n_embd=int(payload.get("n_embd", 256)),
        dropout=float(payload.get("dropout", 0.1)),
        bias=bool(payload.get("bias", True)),
        tokenizer_vocab_size=int(payload.get("tokenizer_vocab_size", payload.get("vocab_size", 16000))),
    )


def load_train_config(project_root: str | Path, config_path: str | Path) -> TrainConfig:
    payload = load_yaml(resolve_path(project_root, config_path))
    split_payload = payload.get("train_val_test_split", {})
    data_payload = payload.get("data", {})
    preprocessing_payload = payload.get("preprocessing", {})
    tokenizer_payload = payload.get("tokenizer", {})
    generation_payload = payload.get("generation", {})
    return TrainConfig(
        seed=int(payload.get("seed", 42)),
        batch_size=int(payload.get("batch_size", 16)),
        gradient_accumulation_steps=int(payload.get("gradient_accumulation_steps", 4)),
        max_steps=int(payload.get("max_steps", 5000)),
        learning_rate=float(payload.get("learning_rate", 3e-4)),
        min_lr=float(payload.get("min_lr", 3e-5)),
        warmup_steps=int(payload.get("warmup_steps", 200)),
        eval_interval=int(payload.get("eval_interval", 250)),
        save_interval=int(payload.get("save_interval", 500)),
        max_grad_norm=float(payload.get("max_grad_norm", 1.0)),
        weight_decay=float(payload.get("weight_decay", 0.1)),
        num_workers=int(payload.get("num_workers", 0)),
        train_val_test_split=SplitConfig(**split_payload),
        data=DataPathsConfig(**data_payload),
        preprocessing=PreprocessingConfig(**preprocessing_payload),
        tokenizer=dict(tokenizer_payload),
        generation=dict(generation_payload),
    )


def setup_logging(name: str, log_file: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        ensure_parent(log_file)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def save_json(payload: Any, output_path: str | Path) -> None:
    ensure_parent(output_path)
    with Path(output_path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def read_jsonl(input_path: str | Path) -> list[dict[str, Any]]:
    path_obj = Path(input_path)
    if not path_obj.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path_obj.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path_obj} at line {line_number}: {exc}") from exc
    return rows


def write_jsonl(records: Iterable[dict[str, Any]], output_path: str | Path) -> None:
    ensure_parent(output_path)
    with Path(output_path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_text(text: str, output_path: str | Path) -> None:
    ensure_parent(output_path)
    Path(output_path).write_text(text, encoding="utf-8")


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def detect_device() -> str:
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m {remaining_seconds}s"
    hours = int(minutes // 60)
    remaining_minutes = minutes % 60
    return f"{hours}h {remaining_minutes}m {remaining_seconds}s"


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", "" if text is None else str(text)).strip()


def chunk_text(text: str, min_chars: int, max_chars: int) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", cleaned) if paragraph.strip()]
    if not paragraphs:
        paragraphs = [cleaned]

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current and len(current) >= min_chars:
            chunks.append(current)
        if len(paragraph) <= max_chars:
            current = paragraph
        else:
            for start in range(0, len(paragraph), max_chars):
                piece = paragraph[start : start + max_chars].strip()
                if len(piece) >= min_chars:
                    chunks.append(piece)
            current = ""
    if current and len(current) >= min_chars:
        chunks.append(current)
    return chunks


def format_dialogue_sample(question: str, answer: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    return (
        "<|system|>\n"
        f"{system_prompt}\n"
        "<|user|>\n"
        f"{question.strip()}\n"
        "<|assistant|>\n"
        f"{answer.strip()}\n"
        "<|endoftext|>\n"
    )


def load_tokenizer_bundle(project_root: str | Path, tokenizer_dir: str | Path) -> TokenizerBundle:
    resolved_dir = resolve_path(project_root, tokenizer_dir)
    tokenizer_path = resolved_dir / "tokenizer.json"
    special_tokens_path = resolved_dir / "special_tokens_map.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
    if not special_tokens_path.exists():
        raise FileNotFoundError(f"Special tokens map not found: {special_tokens_path}")
    with special_tokens_path.open("r", encoding="utf-8") as handle:
        special_tokens = json.load(handle)
    return TokenizerBundle(tokenizer_path=tokenizer_path, special_tokens=special_tokens)


def safe_train_val_test_split(items: list[Any], split_config: SplitConfig) -> tuple[list[Any], list[Any], list[Any]]:
    if not items:
        return [], [], []
    if len(items) == 1:
        return items[:], items[:], items[:]
    if len(items) == 2:
        return [items[0]], [items[1]], [items[1]]

    total = len(items)
    train_end = max(1, int(total * split_config.train))
    val_end = max(train_end + 1, train_end + int(total * split_config.val))
    train_items = items[:train_end]
    val_items = items[train_end:val_end] or items[:1]
    test_items = items[val_end:] or items[-1:]
    return train_items, val_items, test_items
