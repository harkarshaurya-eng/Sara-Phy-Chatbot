from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch
from tokenizers import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import GPTConfig, GPTLanguageModel
from src.utils import (
    SYSTEM_PROMPT,
    detect_device,
    load_tokenizer_bundle,
    load_train_config,
    setup_logging,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a scratch-trained physics GPT model.")
    parser.add_argument("--checkpoint", default="checkpoints/final_model.pt", help="Path to the saved checkpoint.")
    parser.add_argument("--train-config", default="configs/train_config.yaml", help="Training config YAML path.")
    parser.add_argument("--prompt", required=True, help="Prompt text to complete.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature.")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling cutoff.")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p sampling cutoff.")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Maximum number of new tokens.")
    return parser.parse_args()


def build_chat_prompt(user_prompt: str, history: list[tuple[str, str]] | None = None, system_prompt: str = SYSTEM_PROMPT) -> str:
    parts = [f"<|system|>\n{system_prompt}\n"]
    for user_message, assistant_message in history or []:
        parts.append(f"<|user|>\n{user_message.strip()}\n")
        parts.append(f"<|assistant|>\n{assistant_message.strip()}\n")
    parts.append(f"<|user|>\n{user_prompt.strip()}\n")
    parts.append("<|assistant|>\n")
    return "".join(parts)


def load_model_and_tokenizer(checkpoint_path: str | Path, train_config_path: str | Path):
    train_cfg = load_train_config(PROJECT_ROOT, train_config_path)
    tokenizer_bundle = load_tokenizer_bundle(PROJECT_ROOT, train_cfg.data.tokenizer_dir)
    tokenizer = Tokenizer.from_file(str(tokenizer_bundle.tokenizer_path))

    device = detect_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = GPTConfig(**checkpoint["model_config"])
    model = GPTLanguageModel(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, tokenizer, tokenizer_bundle.special_tokens, train_cfg, device


@torch.no_grad()
def generate_response_text(
    model: GPTLanguageModel,
    tokenizer: Tokenizer,
    special_tokens: dict[str, int],
    prompt_text: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    device: str,
) -> str:
    encoded = tokenizer.encode(prompt_text)
    input_ids = encoded.ids[-model.config.block_size :]
    idx = torch.tensor([input_ids], dtype=torch.long, device=device)
    output_ids = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=special_tokens.get("<|endoftext|>"),
    )[0].tolist()

    generated_ids = output_ids[len(input_ids) :]
    if special_tokens.get("<|endoftext|>") in generated_ids:
        eos_index = generated_ids.index(special_tokens["<|endoftext|>"])
        generated_ids = generated_ids[:eos_index]
    text = tokenizer.decode(generated_ids)
    return text.strip()


def main() -> None:
    args = parse_args()
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging("generate", log_file=log_dir / "generate.log")

    model, tokenizer, special_tokens, train_cfg, device = load_model_and_tokenizer(
        checkpoint_path=args.checkpoint,
        train_config_path=args.train_config,
    )
    generation_cfg = train_cfg.generation
    prompt_text = build_chat_prompt(args.prompt)
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
    logger.info("Prompt: %s", args.prompt)
    logger.info("Response: %s", response)
    print(json.dumps({"prompt": args.prompt, "response": response}, indent=2))


if __name__ == "__main__":
    main()
