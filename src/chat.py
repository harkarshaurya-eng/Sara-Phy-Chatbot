from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generate import build_chat_prompt, generate_response_text, load_model_and_tokenizer
from src.utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a terminal chatbot using the scratch-trained physics GPT.")
    parser.add_argument("--checkpoint", default="checkpoints/final_model.pt", help="Path to the saved checkpoint.")
    parser.add_argument("--train-config", default="configs/train_config.yaml", help="Training config YAML path.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature.")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling cutoff.")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p sampling cutoff.")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Maximum number of new tokens.")
    parser.add_argument("--max-history-turns", type=int, default=3, help="How many previous turns to keep.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging("chat", log_file=log_dir / "chat.log")

    model, tokenizer, special_tokens, train_cfg, device = load_model_and_tokenizer(
        checkpoint_path=args.checkpoint,
        train_config_path=args.train_config,
    )
    generation_cfg = train_cfg.generation
    history: list[tuple[str, str]] = []

    print("PhysicsGPT From Scratch terminal chat. Type :quit to exit or :reset to clear history.")
    while True:
        try:
            user_message = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chat.")
            break

        if not user_message:
            continue
        if user_message.lower() in {":quit", ":exit"}:
            break
        if user_message.lower() == ":reset":
            history.clear()
            print("History cleared.")
            continue

        prompt_text = build_chat_prompt(user_message, history=history[-args.max_history_turns :])
        response = generate_response_text(
            model=model,
            tokenizer=tokenizer,
            special_tokens=special_tokens,
            prompt_text=prompt_text,
            max_new_tokens=int(args.max_new_tokens or generation_cfg.get("max_new_tokens", 200)),
            temperature=float(args.temperature if args.temperature is not None else generation_cfg.get("temperature", 0.8)),
            top_k=args.top_k if args.top_k is not None else int(generation_cfg.get("top_k", 50)),
            top_p=args.top_p if args.top_p is not None else float(generation_cfg.get("top_p", 0.95)),
            device=device,
        )
        history.append((user_message, response))
        logger.info("USER: %s", user_message)
        logger.info("ASSISTANT: %s", response)
        print(f"\nPhysicsGPT: {response}")


if __name__ == "__main__":
    main()
