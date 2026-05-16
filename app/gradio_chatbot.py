from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generate import build_chat_prompt, generate_response_text, load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the PhysicsGPT From Scratch Gradio chatbot.")
    parser.add_argument("--checkpoint", default="checkpoints/final_model.pt", help="Path to the saved checkpoint.")
    parser.add_argument("--train-config", default="configs/train_config.yaml", help="Training config YAML path.")
    parser.add_argument("--share", action="store_true", help="Enable Gradio public sharing, useful in Colab.")
    parser.add_argument("--max-history-turns", type=int, default=3, help="How many previous turns to keep.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import gradio as gr
    except ModuleNotFoundError as exc:
        raise RuntimeError("Gradio is not installed. Run `pip install -r requirements.txt` first.") from exc

    model, tokenizer, special_tokens, train_cfg, device = load_model_and_tokenizer(
        checkpoint_path=args.checkpoint,
        train_config_path=args.train_config,
    )
    generation_cfg = train_cfg.generation

    def chat_fn(message: str, history: list[tuple[str, str]]) -> str:
        prompt_text = build_chat_prompt(message, history=history[-args.max_history_turns :])
        return generate_response_text(
            model=model,
            tokenizer=tokenizer,
            special_tokens=special_tokens,
            prompt_text=prompt_text,
            max_new_tokens=int(generation_cfg.get("max_new_tokens", 200)),
            temperature=float(generation_cfg.get("temperature", 0.8)),
            top_k=int(generation_cfg.get("top_k", 50)),
            top_p=float(generation_cfg.get("top_p", 0.95)),
            device=device,
        )

    examples = [
        "Explain Newton's second law.",
        "What is Gauss's law?",
        "Solve: A ball is thrown upward with velocity 20 m/s. Find maximum height.",
        "Explain kinetic energy.",
        "What is entropy?",
    ]

    demo = gr.ChatInterface(
        fn=chat_fn,
        title="PhysicsGPT From Scratch",
        description="A small educational GPT-style physics chatbot trained from scratch.",
        examples=examples,
    )
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
