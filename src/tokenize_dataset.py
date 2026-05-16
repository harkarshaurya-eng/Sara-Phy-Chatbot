from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from tokenizers import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import ensure_dir, load_model_config, load_tokenizer_bundle, load_train_config, read_text, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize train/val/test text files into fixed-length blocks.")
    parser.add_argument("--config", default="configs/tiny_gpt.yaml", help="Model size config YAML path.")
    parser.add_argument("--train-config", default="configs/train_config.yaml", help="Train config YAML path.")
    return parser.parse_args()


def build_blocks(token_ids: list[int], block_size: int, stride: int, pad_id: int) -> torch.Tensor:
    if not token_ids:
        return torch.full((1, block_size + 1), fill_value=pad_id, dtype=torch.long)
    if len(token_ids) <= block_size:
        padded = token_ids + [pad_id] * (block_size + 1 - len(token_ids))
        return torch.tensor([padded], dtype=torch.long)

    blocks: list[list[int]] = []
    for start in range(0, max(1, len(token_ids) - block_size), stride):
        end = start + block_size + 1
        if end > len(token_ids):
            tail = token_ids[-(block_size + 1) :]
            if len(tail) < block_size + 1:
                tail = tail + [pad_id] * (block_size + 1 - len(tail))
            blocks.append(tail)
            break
        blocks.append(token_ids[start:end])
    return torch.tensor(blocks, dtype=torch.long)


def encode_file(tokenizer: Tokenizer, input_path: Path) -> list[int]:
    text = read_text(input_path)
    return tokenizer.encode(text).ids


def main() -> None:
    args = parse_args()
    model_cfg = load_model_config(PROJECT_ROOT, args.config)
    train_cfg = load_train_config(PROJECT_ROOT, args.train_config)
    log_dir = ensure_dir(PROJECT_ROOT / "logs")
    logger = setup_logging("tokenize_dataset", log_file=log_dir / "tokenize_dataset.log")

    tokenizer_bundle = load_tokenizer_bundle(PROJECT_ROOT, train_cfg.data.tokenizer_dir)
    tokenizer = Tokenizer.from_file(str(tokenizer_bundle.tokenizer_path))
    block_size = int(model_cfg.block_size)
    stride = max(1, int(block_size * float(train_cfg.preprocessing.stride_ratio)))
    pad_id = int(tokenizer_bundle.special_tokens["<|pad|>"])

    split_files = {
        "train": PROJECT_ROOT / train_cfg.data.train_text_file,
        "val": PROJECT_ROOT / train_cfg.data.val_text_file,
        "test": PROJECT_ROOT / train_cfg.data.test_text_file,
    }
    output_files = {
        "train": PROJECT_ROOT / train_cfg.data.train_tokens_file,
        "val": PROJECT_ROOT / train_cfg.data.val_tokens_file,
        "test": PROJECT_ROOT / train_cfg.data.test_tokens_file,
    }

    for split_name, input_path in split_files.items():
        if not input_path.exists():
            raise FileNotFoundError(f"Missing text split file: {input_path}")
        token_ids = encode_file(tokenizer, input_path=input_path)
        blocks = build_blocks(token_ids, block_size=block_size, stride=stride, pad_id=pad_id)
        torch.save(
            {
                "blocks": blocks,
                "block_size": block_size,
                "pad_token_id": pad_id,
                "token_count": len(token_ids),
            },
            output_files[split_name],
        )
        logger.info("Saved %s split with %s token IDs and %s blocks to %s", split_name, len(token_ids), len(blocks), output_files[split_name])

    print({"block_size": block_size, "stride": stride, "train_tokens_file": str(output_files["train"])})


if __name__ == "__main__":
    main()
