from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generate import build_chat_prompt, generate_response_text, load_model_and_tokenizer
from src.utils import normalize_text, read_jsonl, setup_logging


NUMBER_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight evaluation on held-out physics examples.")
    parser.add_argument("--checkpoint", default="checkpoints/final_model.pt", help="Path to the saved checkpoint.")
    parser.add_argument("--train-config", default="configs/train_config.yaml", help="Training config YAML path.")
    parser.add_argument("--max-samples", type=int, default=20, help="How many held-out records to evaluate.")
    return parser.parse_args()


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).lower().split()
    ref_tokens = normalize_text(reference).lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = 0
    ref_counts: dict[str, int] = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    for token in pred_tokens:
        if ref_counts.get(token, 0) > 0:
            common += 1
            ref_counts[token] -= 1
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def extract_last_number(text: str) -> float | None:
    matches = NUMBER_PATTERN.findall(normalize_text(text))
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def main() -> None:
    args = parse_args()
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging("evaluate", log_file=log_dir / "evaluate.log")

    model, tokenizer, special_tokens, train_cfg, device = load_model_and_tokenizer(
        checkpoint_path=args.checkpoint,
        train_config_path=args.train_config,
    )
    records_path = PROJECT_ROOT / train_cfg.data.processed_dir / "test_records.jsonl"
    if not records_path.exists():
        raise FileNotFoundError(f"Missing evaluation records: {records_path}. Run src/prepare_text_corpus.py first.")

    examples = read_jsonl(records_path)[: args.max_samples]
    if not examples:
        raise RuntimeError("No evaluation examples found in the test split.")

    exact_matches = 0
    f1_scores: list[float] = []
    numeric_squared_errors: list[float] = []
    numeric_absolute_errors: list[float] = []
    outputs: list[dict[str, object]] = []
    for example in examples:
        prompt_text = build_chat_prompt(example["question"])
        prediction = generate_response_text(
            model=model,
            tokenizer=tokenizer,
            special_tokens=special_tokens,
            prompt_text=prompt_text,
            max_new_tokens=int(train_cfg.generation.get("max_new_tokens", 200)),
            temperature=0.7,
            top_k=int(train_cfg.generation.get("top_k", 50)),
            top_p=float(train_cfg.generation.get("top_p", 0.95)),
            device=device,
        )
        reference = example["answer"]
        exact = int(normalize_text(prediction).lower() == normalize_text(reference).lower())
        exact_matches += exact
        f1 = token_f1(prediction, reference)
        f1_scores.append(f1)
        prediction_value = extract_last_number(prediction)
        reference_value = extract_last_number(reference)
        squared_error = None
        absolute_error = None
        if prediction_value is not None and reference_value is not None:
            absolute_error = abs(prediction_value - reference_value)
            squared_error = absolute_error ** 2
            numeric_absolute_errors.append(absolute_error)
            numeric_squared_errors.append(squared_error)
        outputs.append(
            {
                "topic": example.get("topic", "general physics"),
                "question": example["question"],
                "prediction": prediction,
                "reference": reference,
                "exact_match": exact,
                "token_f1": f1,
                "prediction_value": prediction_value,
                "reference_value": reference_value,
                "absolute_error": absolute_error,
                "squared_error": squared_error,
            }
        )

    mse = None
    rmse = None
    mae = None
    if numeric_squared_errors:
        mse = sum(numeric_squared_errors) / len(numeric_squared_errors)
        rmse = math.sqrt(mse)
        mae = sum(numeric_absolute_errors) / len(numeric_absolute_errors)

    summary = {
        "examples": len(outputs),
        "exact_match": exact_matches / max(1, len(outputs)),
        "token_f1": sum(f1_scores) / max(1, len(f1_scores)),
        "perplexity_proxy": math.exp(min(5.0, 1.0 - min(1.0, sum(f1_scores) / max(1, len(f1_scores))))),
        "numeric_examples": len(numeric_squared_errors),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
    }
    logger.info("Evaluation summary: %s", summary)
    output_path = PROJECT_ROOT / "checkpoints" / "evaluation_report.json"
    output_path.write_text(json.dumps({"summary": summary, "samples": outputs}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"summary": summary, "report": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
