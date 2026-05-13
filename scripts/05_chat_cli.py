from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import interactive_chat, load_chat_model
from src.train_utils import get_system_prompt, load_config, resolve_path, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start a local terminal chat session with the physics chatbot.")
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML config file.")
    parser.add_argument("--adapter", default="outputs/adapters/final", help="LoRA adapter path.")
    parser.add_argument("--merged-model", default=None, help="Use a merged model directory instead of base+adapter.")
    parser.add_argument("--base-model", default=None, help="Optional override for the base model.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=None, help="Sampling top-p value.")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Maximum response tokens.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    logger = setup_logging(
        "chat_cli",
        log_file=resolve_path(PROJECT_ROOT, config.get("log_dir", "outputs/logs")) / "chat_cli.log",
    )

    base_model = args.base_model or config.get("base_model")
    merged_model_path = resolve_path(PROJECT_ROOT, args.merged_model) if args.merged_model else None
    adapter_path = None if merged_model_path else resolve_path(PROJECT_ROOT, args.adapter)
    if adapter_path and not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

    model, tokenizer, runtime = load_chat_model(
        base_model_name=str(base_model),
        adapter_path=str(adapter_path) if adapter_path else None,
        merged_model_path=str(merged_model_path) if merged_model_path else None,
        trust_remote_code=bool(config.get("trust_remote_code", True)),
        logger=logger,
    )

    logger.info("Chat runtime: %s", runtime)
    inference_cfg = config.get("inference", {})
    interactive_chat(
        model=model,
        tokenizer=tokenizer,
        model_name=str(base_model),
        system_prompt=get_system_prompt(config),
        temperature=args.temperature if args.temperature is not None else float(inference_cfg.get("temperature", 0.7)),
        top_p=args.top_p if args.top_p is not None else float(inference_cfg.get("top_p", 0.9)),
        max_new_tokens=(
            args.max_new_tokens if args.max_new_tokens is not None else int(inference_cfg.get("max_new_tokens", 512))
        ),
    )


if __name__ == "__main__":
    main()
