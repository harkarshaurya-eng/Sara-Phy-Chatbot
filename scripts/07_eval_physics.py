from __future__ import annotations

import argparse
import math
import re
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import generate_chat_response, load_chat_model
from src.train_utils import get_system_prompt, load_config, read_jsonl, resolve_path, setup_logging, write_text


DOMAIN_PROBES = [
    {
        "topic": "mechanics",
        "prompt": "Explain Newton's second law and define every symbol in the equation.",
        "must_include": ["F", "m", "a", "newton"],
    },
    {
        "topic": "electromagnetism",
        "prompt": "State Ohm's law and explain the SI units of voltage, current, and resistance.",
        "must_include": ["V", "I", "R", "volt", "ampere", "ohm"],
    },
    {
        "topic": "thermodynamics",
        "prompt": "What does the first law of thermodynamics mean in physics?",
        "must_include": ["energy", "heat", "work"],
    },
    {
        "topic": "optics",
        "prompt": "Explain refraction and mention Snell's law.",
        "must_include": ["refraction", "snell", "index"],
    },
    {
        "topic": "waves",
        "prompt": "Relate wave speed, frequency, and wavelength.",
        "must_include": ["v", "f", "lambda"],
    },
    {
        "topic": "relativity",
        "prompt": "What is time dilation in special relativity?",
        "must_include": ["relativity", "observer", "speed of light"],
    },
    {
        "topic": "quantum physics",
        "prompt": "Explain the uncertainty principle.",
        "must_include": ["uncertainty", "position", "momentum"],
    },
    {
        "topic": "nuclear physics",
        "prompt": "What is radioactive half-life?",
        "must_include": ["half-life", "decay"],
    },
    {
        "topic": "astrophysics",
        "prompt": "Explain how a main-sequence star produces energy.",
        "must_include": ["fusion", "hydrogen", "helium"],
    },
    {
        "topic": "mathematical physics",
        "prompt": "What is the difference between a Lagrangian and a Hamiltonian?",
        "must_include": ["lagrangian", "hamiltonian"],
    },
    {
        "topic": "experimental physics",
        "prompt": "Why are measurement uncertainty and error bars important in experiments?",
        "must_include": ["uncertainty", "measurement", "error"],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight evaluation pass for the physics chatbot.")
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML config file.")
    parser.add_argument("--adapter", default="outputs/adapters/final", help="Path to the LoRA adapter.")
    parser.add_argument("--merged-model", default=None, help="Use a merged model instead of base+adapter.")
    parser.add_argument("--base-model", default=None, help="Optional base model override.")
    parser.add_argument("--max-samples", type=int, default=16, help="How many held-out samples to score.")
    return parser.parse_args()


def normalize_eval_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_eval_text(prediction).split()
    ref_tokens = normalize_eval_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = 0
    ref_counts = {}
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


NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def extract_numeric_candidates(text: str) -> list[float]:
    values: list[float] = []
    for match in NUMBER_RE.findall(text.replace(",", "")):
        try:
            values.append(float(match))
        except ValueError:
            continue
    return values


def extract_scalar_target(text: str) -> float | None:
    values = extract_numeric_candidates(text)
    if not values:
        return None
    return values[-1]


def compute_numeric_regression_metrics(rows: list[dict]) -> dict | None:
    if not rows:
        return None

    actual_values = [row["reference_value"] for row in rows]
    predicted_values = [row["prediction_value"] for row in rows]
    mean_actual = statistics.mean(actual_values)
    squared_errors = [(pred - actual) ** 2 for pred, actual in zip(predicted_values, actual_values)]
    absolute_errors = [abs(pred - actual) for pred, actual in zip(predicted_values, actual_values)]
    denominator = sum(abs(actual - mean_actual) for actual in actual_values)

    return {
        "count": len(rows),
        "rmse": math.sqrt(sum(squared_errors) / len(squared_errors)),
        "mae": sum(absolute_errors) / len(absolute_errors),
        "rae": (sum(absolute_errors) / denominator) if denominator > 0 else 0.0,
    }


def extract_user_and_reference(example: dict) -> tuple[str, str]:
    user_text = ""
    assistant_text = ""
    for message in example.get("messages", []):
        if message.get("role") == "user" and not user_text:
            user_text = str(message.get("content", ""))
        if message.get("role") == "assistant" and not assistant_text:
            assistant_text = str(message.get("content", ""))
    return user_text, assistant_text


def main() -> None:
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    logger = setup_logging(
        "eval_physics",
        log_file=resolve_path(PROJECT_ROOT, config.get("log_dir", "outputs/logs")) / "eval_physics.log",
    )

    base_model = str(args.base_model or config.get("base_model"))
    merged_model_path = resolve_path(PROJECT_ROOT, args.merged_model) if args.merged_model else None
    adapter_path = None if merged_model_path else resolve_path(PROJECT_ROOT, args.adapter)
    if adapter_path and not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

    model, tokenizer, runtime = load_chat_model(
        base_model_name=base_model,
        adapter_path=str(adapter_path) if adapter_path else None,
        merged_model_path=str(merged_model_path) if merged_model_path else None,
        trust_remote_code=bool(config.get("trust_remote_code", True)),
        load_in_4bit=bool(config.get("load_in_4bit", False)),
        logger=logger,
    )

    system_prompt = get_system_prompt(config)
    test_split_path = resolve_path(PROJECT_ROOT, config.get("test_split_path", "data/final/test.jsonl"))
    held_out_rows = read_jsonl(test_split_path)[: args.max_samples] if test_split_path.exists() else []

    held_out_scores = []
    numeric_scores = []
    for row in held_out_rows:
        user_text, reference_text = extract_user_and_reference(row)
        if not user_text or not reference_text:
            continue
        result = generate_chat_response(
            model=model,
            tokenizer=tokenizer,
            model_name=base_model,
            messages=[{"role": "user", "content": user_text}],
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=min(512, int(config.get("inference", {}).get("max_new_tokens", 512))),
            system_prompt=system_prompt,
        )
        score = token_f1(result["text"], reference_text)
        held_out_scores.append(
            {
                "topic": row.get("topic", "general physics"),
                "prompt": user_text,
                "reference": reference_text[:300],
                "prediction": result["text"][:300],
                "token_f1": score,
            }
        )
        reference_value = extract_scalar_target(reference_text)
        prediction_value = extract_scalar_target(result["text"])
        if reference_value is not None and prediction_value is not None:
            numeric_scores.append(
                {
                    "topic": row.get("topic", "general physics"),
                    "prompt": user_text,
                    "reference_value": reference_value,
                    "prediction_value": prediction_value,
                    "absolute_error": abs(prediction_value - reference_value),
                }
            )

    probe_results = []
    for probe in DOMAIN_PROBES:
        result = generate_chat_response(
            model=model,
            tokenizer=tokenizer,
            model_name=base_model,
            messages=[{"role": "user", "content": probe["prompt"]}],
            temperature=0.2,
            top_p=0.95,
            max_new_tokens=384,
            system_prompt=system_prompt,
        )
        generated = normalize_eval_text(result["text"])
        matched = [keyword for keyword in probe["must_include"] if keyword.lower() in generated]
        probe_results.append(
            {
                "topic": probe["topic"],
                "prompt": probe["prompt"],
                "matched": matched,
                "coverage": len(matched) / max(1, len(probe["must_include"])),
                "response": result["text"][:400],
            }
        )

    held_out_mean = statistics.mean(row["token_f1"] for row in held_out_scores) if held_out_scores else 0.0
    probe_mean = statistics.mean(row["coverage"] for row in probe_results) if probe_results else 0.0
    numeric_metrics = compute_numeric_regression_metrics(numeric_scores)
    report_lines = [
        "# Physics Evaluation Report",
        f"- Base model: `{base_model}`",
        f"- Runtime: `{runtime.get('accelerator')}`",
        f"- Held-out sample count: `{len(held_out_scores)}`",
        f"- Mean held-out token F1: `{held_out_mean:.4f}`",
        f"- Mean domain probe coverage: `{probe_mean:.4f}`",
        f"- Numeric subset count: `{numeric_metrics['count'] if numeric_metrics else 0}`",
        f"- Numeric RMSE: `{numeric_metrics['rmse']:.6f}`" if numeric_metrics else "- Numeric RMSE: `n/a`",
        f"- Numeric RAE: `{numeric_metrics['rae']:.6f}`" if numeric_metrics else "- Numeric RAE: `n/a`",
        f"- Numeric MAE: `{numeric_metrics['mae']:.6f}`" if numeric_metrics else "- Numeric MAE: `n/a`",
        "",
        "## Held-Out Sample Scores",
    ]

    if held_out_scores:
        for row in held_out_scores:
            report_lines.extend(
                [
                    f"### {row['topic']}",
                    f"- Prompt: {row['prompt']}",
                    f"- Token F1: {row['token_f1']:.4f}",
                    f"- Reference excerpt: {row['reference']}",
                    f"- Prediction excerpt: {row['prediction']}",
                    "",
                ]
            )
    else:
        report_lines.append("No held-out test split was found, so only domain probes were run.\n")

    report_lines.append("## Domain Probe Checks")
    for row in probe_results:
        report_lines.extend(
            [
                f"### {row['topic']}",
                f"- Prompt: {row['prompt']}",
                f"- Keyword coverage: {row['coverage']:.4f}",
                f"- Matched keywords: {', '.join(row['matched']) if row['matched'] else 'none'}",
                f"- Response excerpt: {row['response']}",
                "",
            ]
        )

    report_lines.append("## Numeric Regression Metrics")
    if numeric_metrics:
        report_lines.extend(
            [
                "RMSE and RAE are computed only on held-out examples where both the reference and generated answer contain at least one numeric value.",
                "The evaluator uses the last numeric value in each text as a scalar target. This is a practical heuristic for final-answer style physics problems, not a full symbolic grader.",
                "",
            ]
        )
        for row in numeric_scores[: min(10, len(numeric_scores))]:
            report_lines.extend(
                [
                    f"### {row['topic']}",
                    f"- Prompt: {row['prompt']}",
                    f"- Reference value: {row['reference_value']}",
                    f"- Prediction value: {row['prediction_value']}",
                    f"- Absolute error: {row['absolute_error']}",
                    "",
                ]
            )
    else:
        report_lines.extend(
            [
                "No numeric subset could be extracted from the evaluated samples, so RMSE and RAE were not computed.",
                "",
            ]
        )

    report_lines.extend(
        [
            "## Qualitative Scoring Prompt Template",
            "Use the template below with a separate evaluator if you want a second-pass review of pedagogy and factual caution:",
            "",
            "```text",
            "You are grading a physics tutor answer.",
            "Score 1-5 for factual accuracy, conceptual clarity, step-by-step reasoning, formula correctness, SI unit usage, and honesty about uncertainty.",
            "Return a short rubric-based critique and one concrete improvement.",
            "```",
        ]
    )

    report_path = resolve_path(PROJECT_ROOT, config.get("log_dir", "outputs/logs")) / "eval_report.md"
    write_text("\n".join(report_lines), report_path)
    logger.info("Saved evaluation report to %s", report_path)
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
