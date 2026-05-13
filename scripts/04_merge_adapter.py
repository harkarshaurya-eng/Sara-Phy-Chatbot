from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import attach_adapter, load_base_model, load_tokenizer_for_model
from src.train_utils import ensure_dir, load_config, resolve_path, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into the base model for standalone inference.")
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML config file.")
    parser.add_argument("--adapter", default="outputs/adapters/final", help="Path to the trained adapter directory.")
    parser.add_argument("--output-dir", default="outputs/merged_model", help="Where to save the merged model.")
    parser.add_argument("--base-model", default=None, help="Optional override for the base model name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    logger = setup_logging(
        "merge_adapter",
        log_file=resolve_path(PROJECT_ROOT, config.get("log_dir", "outputs/logs")) / "merge_adapter.log",
    )

    base_model = args.base_model or config.get("base_model")
    adapter_path = resolve_path(PROJECT_ROOT, args.adapter)
    output_dir = ensure_dir(resolve_path(PROJECT_ROOT, args.output_dir))
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_path}")

    tokenizer = load_tokenizer_for_model(str(adapter_path), trust_remote_code=bool(config.get("trust_remote_code", True)))
    base_model_obj, _ = load_base_model(
        str(base_model),
        trust_remote_code=bool(config.get("trust_remote_code", True)),
        prefer_bf16=bool(config.get("bf16", True)),
        logger=logger,
    )
    peft_model = attach_adapter(base_model_obj, adapter_path=adapter_path, logger=logger)
    merged = peft_model.merge_and_unload()

    merged.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Merged model saved to %s", output_dir)


if __name__ == "__main__":
    main()
