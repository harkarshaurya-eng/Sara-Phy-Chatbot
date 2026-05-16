from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    SYSTEM_PROMPT,
    chunk_text,
    ensure_dir,
    format_dialogue_sample,
    load_train_config,
    normalize_text,
    read_jsonl,
    safe_train_val_test_split,
    seed_everything,
    setup_logging,
    write_jsonl,
    write_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a plain text corpus for from-scratch GPT training.")
    parser.add_argument("--train-config", default="configs/train_config.yaml", help="Training config YAML path.")
    return parser.parse_args()


def infer_topic(record: dict[str, Any]) -> str:
    topic = normalize_text(record.get("topic", ""))
    return topic or "general physics"


def raw_record_to_training_example(record: dict[str, Any], min_text_chars: int, max_text_chars: int) -> list[dict[str, str]]:
    record_type = record.get("record_type")
    examples: list[dict[str, str]] = []

    if record_type == "qa":
        question = normalize_text(record.get("question"))
        answer = normalize_text(record.get("answer"))
        if question and answer:
            examples.append(
                {
                    "source": normalize_text(record.get("source", "unknown")),
                    "topic": infer_topic(record),
                    "question": question,
                    "answer": answer,
                }
            )
        return examples

    if record_type == "text":
        title = normalize_text(record.get("title", "physics concept"))
        text = normalize_text(record.get("text"))
        for chunk in chunk_text(text, min_chars=min_text_chars, max_chars=max_text_chars):
            examples.append(
                {
                    "source": normalize_text(record.get("source", "unknown")),
                    "topic": infer_topic(record),
                    "question": f"Explain this physics topic clearly: {title}",
                    "answer": chunk,
                }
            )
        return examples

    return examples


def is_low_quality(question: str, answer: str, min_question_chars: int, min_answer_chars: int) -> bool:
    if not question or not answer:
        return True
    if len(question) < min_question_chars or len(answer) < min_answer_chars:
        return True
    return False


def load_raw_records(raw_dir: Path, sample_records_path: Path, logger) -> list[dict[str, Any]]:
    raw_records: list[dict[str, Any]] = []
    for file_path in sorted(raw_dir.glob("*.jsonl")):
        logger.info("Loading raw file %s", file_path)
        raw_records.extend(read_jsonl(file_path))
    if not raw_records and sample_records_path.exists():
        logger.warning("No raw records found. Falling back to sample records.")
        raw_records.extend(read_jsonl(sample_records_path))
    return raw_records


def main() -> None:
    args = parse_args()
    train_cfg = load_train_config(PROJECT_ROOT, args.train_config)
    seed_everything(train_cfg.seed)
    log_dir = ensure_dir(PROJECT_ROOT / "logs")
    logger = setup_logging("prepare_text_corpus", log_file=log_dir / "prepare_text_corpus.log")

    raw_dir = PROJECT_ROOT / train_cfg.data.raw_dir
    processed_dir = ensure_dir(PROJECT_ROOT / train_cfg.data.processed_dir)
    sample_records_path = PROJECT_ROOT / train_cfg.data.sample_records_file
    raw_records = load_raw_records(raw_dir, sample_records_path=sample_records_path, logger=logger)
    logger.info("Loaded %s raw records.", len(raw_records))

    examples: list[dict[str, str]] = []
    for raw_record in raw_records:
        examples.extend(
            raw_record_to_training_example(
                raw_record,
                min_text_chars=train_cfg.preprocessing.min_text_chars,
                max_text_chars=train_cfg.preprocessing.max_text_chars,
            )
        )

    cleaned: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for example in examples:
        question = normalize_text(example["question"])
        answer = normalize_text(example["answer"])
        if is_low_quality(
            question,
            answer,
            min_question_chars=train_cfg.preprocessing.min_question_chars,
            min_answer_chars=train_cfg.preprocessing.min_answer_chars,
        ):
            continue
        key = (question.lower(), answer.lower())
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(
            {
                "source": example["source"],
                "topic": example["topic"],
                "question": question,
                "answer": answer,
                "text": format_dialogue_sample(question, answer, system_prompt=SYSTEM_PROMPT),
            }
        )

    if not cleaned:
        raise RuntimeError("No usable training examples were produced. Check data/raw or the sample records.")

    random.shuffle(cleaned)
    train_examples, val_examples, test_examples = safe_train_val_test_split(cleaned, train_cfg.train_val_test_split)

    corpus_text = "\n".join(example["text"] for example in cleaned)
    train_text = "\n".join(example["text"] for example in train_examples)
    val_text = "\n".join(example["text"] for example in val_examples)
    test_text = "\n".join(example["text"] for example in test_examples)

    write_text(corpus_text, PROJECT_ROOT / train_cfg.data.corpus_file)
    write_text(train_text, PROJECT_ROOT / train_cfg.data.train_text_file)
    write_text(val_text, PROJECT_ROOT / train_cfg.data.val_text_file)
    write_text(test_text, PROJECT_ROOT / train_cfg.data.test_text_file)

    write_jsonl(train_examples, processed_dir / "train_records.jsonl")
    write_jsonl(val_examples, processed_dir / "val_records.jsonl")
    write_jsonl(test_examples, processed_dir / "test_records.jsonl")

    logger.info("Saved corpus to %s", PROJECT_ROOT / train_cfg.data.corpus_file)
    logger.info("Train/val/test example counts: %s / %s / %s", len(train_examples), len(val_examples), len(test_examples))
    print(
        {
            "train_examples": len(train_examples),
            "val_examples": len(val_examples),
            "test_examples": len(test_examples),
            "corpus_path": str(PROJECT_ROOT / train_cfg.data.corpus_file),
        }
    )


if __name__ == "__main__":
    main()
