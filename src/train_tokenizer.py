from __future__ import annotations

import argparse
from pathlib import Path
import sys

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import ensure_dir, load_model_config, load_train_config, save_json, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer from scratch.")
    parser.add_argument("--config", default="configs/tiny_gpt.yaml", help="Model size config YAML path.")
    parser.add_argument("--train-config", default="configs/train_config.yaml", help="Train config YAML path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_cfg = load_model_config(PROJECT_ROOT, args.config)
    train_cfg = load_train_config(PROJECT_ROOT, args.train_config)
    log_dir = ensure_dir(PROJECT_ROOT / "logs")
    logger = setup_logging("train_tokenizer", log_file=log_dir / "train_tokenizer.log")

    corpus_path = PROJECT_ROOT / train_cfg.data.corpus_file
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}. Run src/prepare_text_corpus.py first.")

    tokenizer_dir = ensure_dir(PROJECT_ROOT / train_cfg.data.tokenizer_dir)
    special_tokens = list(train_cfg.tokenizer.get("special_tokens", []))
    if not special_tokens:
        raise ValueError("No special tokens configured in configs/train_config.yaml")

    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=int(model_cfg.tokenizer_vocab_size),
        min_frequency=2,
        special_tokens=special_tokens,
        show_progress=True,
    )
    tokenizer.train([str(corpus_path)], trainer=trainer)

    tokenizer_path = tokenizer_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    vocab_path = tokenizer_dir / "vocab.json"
    merges_dir = tokenizer_dir / "bpe_artifacts"
    ensure_dir(merges_dir)
    tokenizer.model.save(str(merges_dir))

    special_token_map = {token: tokenizer.token_to_id(token) for token in special_tokens}
    save_json(special_token_map, tokenizer_dir / "special_tokens_map.json")
    save_json(
        {
            "model_name": model_cfg.name,
            "requested_vocab_size": model_cfg.tokenizer_vocab_size,
            "actual_vocab_size": tokenizer.get_vocab_size(),
            "tokenizer_path": str(tokenizer_path),
            "vocab_path": str(vocab_path),
        },
        tokenizer_dir / "tokenizer_config.json",
    )

    logger.info("Tokenizer saved to %s", tokenizer_path)
    logger.info("Actual vocab size: %s", tokenizer.get_vocab_size())
    print({"tokenizer_path": str(tokenizer_path), "vocab_size": tokenizer.get_vocab_size()})


if __name__ == "__main__":
    main()
