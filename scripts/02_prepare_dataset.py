from __future__ import annotations

import argparse
import glob
from collections import Counter
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_cleaning import (
    SimHashDeduper,
    exact_dedupe_key,
    is_low_quality,
    normalize_text,
    shuffle_and_split,
    source_counts,
    split_into_text_chunks,
    topic_counts,
)
from src.formatting import DEFAULT_SYSTEM_PROMPT, format_qa_example, format_text_example
from src.train_utils import (
    get_system_prompt,
    load_config,
    markdown_table,
    read_jsonl,
    resolve_path,
    setup_logging,
    write_jsonl,
    write_text,
)

ROLE_ALIASES = {
    "system": "system",
    "user": "user",
    "human": "user",
    "prompter": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "model": "assistant",
    "bot": "assistant",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean, deduplicate, and convert raw datasets into chat-format JSONL.")
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML config file.")
    return parser.parse_args()


def _load_custom_records(custom_glob_pattern: str) -> list[dict]:
    records: list[dict] = []
    for file_path in sorted(glob.glob(custom_glob_pattern)):
        source_name = Path(file_path).stem
        for row in read_jsonl(file_path):
            if "messages" in row:
                row.setdefault("source", source_name)
                row.setdefault("license", "user-provided")
                records.append(row)
                continue

            question = row.get("question") or row.get("prompt") or row.get("instruction") or row.get("input")
            answer = row.get("answer") or row.get("response") or row.get("output") or row.get("assistant")
            text = row.get("text")

            if question and answer:
                records.append(
                    {
                        "record_type": "qa",
                        "source": row.get("source", source_name),
                        "license": row.get("license", "user-provided"),
                        "question": question,
                        "answer": answer,
                        "topic": row.get("topic"),
                        "difficulty": row.get("difficulty"),
                    }
                )
            elif text:
                records.append(
                    {
                        "record_type": "text",
                        "source": row.get("source", source_name),
                        "license": row.get("license", "user-provided"),
                        "title": row.get("title") or row.get("topic") or source_name,
                        "text": text,
                        "topic": row.get("topic"),
                        "difficulty": row.get("difficulty"),
                    }
                )
    return records


def _normalize_conversation_record(record: dict, system_prompt: str) -> dict | None:
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return None

    normalized_messages = []
    for message in messages:
        raw_role = str(message.get("role", "")).strip().lower()
        role = ROLE_ALIASES.get(raw_role, raw_role)
        content = normalize_text(message.get("content", ""))
        if not role or not content:
            continue
        if role not in {"system", "user", "assistant"}:
            continue
        if normalized_messages and normalized_messages[-1]["role"] == role:
            normalized_messages[-1]["content"] = f"{normalized_messages[-1]['content']}\n\n{content}"
        else:
            normalized_messages.append({"role": role, "content": content})

    if not normalized_messages:
        return None
    if normalized_messages[0]["role"] != "system":
        normalized_messages.insert(0, {"role": "system", "content": system_prompt})

    return {
        "messages": normalized_messages,
        "source": record.get("source", "custom"),
        "topic": record.get("topic", "general physics"),
        "difficulty": record.get("difficulty", "intermediate"),
        "license": record.get("license", "user-provided"),
    }


def _record_to_examples(record: dict, config: dict, system_prompt: str) -> list[dict]:
    record_type = record.get("record_type")
    examples: list[dict] = []

    if "messages" in record:
        normalized = _normalize_conversation_record(record, system_prompt=system_prompt)
        return [normalized] if normalized else []

    if record_type == "qa":
        question = normalize_text(record.get("question", ""))
        answer = normalize_text(record.get("answer", ""))
        if is_low_quality(
            question,
            answer,
            min_question_chars=int(config.get("min_question_chars", 10)),
            min_answer_chars=int(config.get("min_answer_chars", 40)),
        ):
            return []
        qa_record = dict(record)
        qa_record["question"] = question
        qa_record["answer"] = answer
        examples.append(format_qa_example(qa_record, system_prompt=system_prompt))
        return examples

    if record_type == "text":
        text = normalize_text(record.get("text", ""))
        chunks = split_into_text_chunks(
            text,
            min_chars=int(config.get("min_text_chunk_chars", 300)),
            max_chars=int(config.get("max_text_chunk_chars", 1800)),
        )
        for chunk in chunks:
            examples.append(format_text_example(record, text_chunk=chunk, system_prompt=system_prompt))
        return examples

    return []


def _example_text(example: dict) -> str:
    return "\n".join(f"{message['role']}: {message['content']}" for message in example["messages"])


def build_report(
    all_examples: list[dict],
    train_split: list[dict],
    validation_split: list[dict],
    test_split: list[dict],
    stats: Counter,
) -> str:
    header = "# Dataset Preparation Report"
    summary = markdown_table(
        ["Metric", "Value"],
        [
            ["Final examples", len(all_examples)],
            ["Train split", len(train_split)],
            ["Validation split", len(validation_split)],
            ["Test split", len(test_split)],
            ["Dropped low-quality", stats["dropped_low_quality"]],
            ["Dropped exact duplicates", stats["dropped_exact_duplicates"]],
            ["Dropped near duplicates", stats["dropped_near_duplicates"]],
        ],
    )

    source_table = markdown_table(
        ["Source", "Count"],
        [[source, count] for source, count in sorted(source_counts(all_examples).items())],
    )
    topic_table = markdown_table(
        ["Topic", "Count"],
        [[topic, count] for topic, count in sorted(topic_counts(all_examples).items())],
    )

    return "\n\n".join(
        [
            header,
            "## Summary\n" + summary,
            "## By Source\n" + source_table,
            "## By Topic\n" + topic_table,
        ]
    )


def main() -> None:
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    log_path = resolve_path(PROJECT_ROOT, config.get("log_dir", "outputs/logs")) / "prepare_dataset.log"
    logger = setup_logging("prepare_dataset", log_file=log_path)

    raw_dir = resolve_path(PROJECT_ROOT, config.get("raw_data_dir", "data/raw"))
    custom_glob_pattern = str(resolve_path(PROJECT_ROOT, config.get("custom_data_glob", "data/custom/*.jsonl")))
    final_dataset_path = resolve_path(PROJECT_ROOT, config.get("final_dataset_path", "data/final/physics_sft.jsonl"))
    train_split_path = resolve_path(PROJECT_ROOT, config.get("train_split_path", "data/final/train.jsonl"))
    validation_split_path = resolve_path(PROJECT_ROOT, config.get("validation_split_path", "data/final/validation.jsonl"))
    test_split_path = resolve_path(PROJECT_ROOT, config.get("test_split_path", "data/final/test.jsonl"))
    report_path = resolve_path(PROJECT_ROOT, config.get("log_dir", "outputs/logs")) / "data_report.md"

    system_prompt = get_system_prompt(config) or DEFAULT_SYSTEM_PROMPT
    stats: Counter = Counter()

    raw_records: list[dict] = []
    for file_path in sorted(raw_dir.glob("*.jsonl")):
        logger.info("Loading raw dataset file %s", file_path)
        raw_records.extend(read_jsonl(file_path))
    raw_records.extend(_load_custom_records(custom_glob_pattern))
    logger.info("Loaded %s raw/custom records before cleaning.", len(raw_records))

    exact_keys: set[str] = set()
    near_deduper = SimHashDeduper(threshold=int(config.get("near_duplicate_hamming_threshold", 3)))
    final_examples: list[dict] = []

    for record in raw_records:
        converted_examples = _record_to_examples(record, config=config, system_prompt=system_prompt)
        if not converted_examples:
            stats["dropped_low_quality"] += 1
            continue

        for example in converted_examples:
            dedupe_key = exact_dedupe_key(example["messages"], source=example["source"])
            if dedupe_key in exact_keys:
                stats["dropped_exact_duplicates"] += 1
                continue

            combined_text = _example_text(example)
            if near_deduper.is_duplicate(combined_text):
                stats["dropped_near_duplicates"] += 1
                continue

            exact_keys.add(dedupe_key)
            final_examples.append(example)

    if not final_examples:
        raise RuntimeError(
            "No training examples were produced. Check dataset downloads, custom JSONL files, or the quality filters."
        )

    train_split, validation_split, test_split = shuffle_and_split(
        final_examples,
        train_ratio=float(config.get("train_ratio", 0.94)),
        validation_ratio=float(config.get("validation_ratio", 0.03)),
        test_ratio=float(config.get("test_ratio", 0.03)),
        seed=int(config.get("seed", 42)),
    )

    write_jsonl(final_examples, final_dataset_path)
    write_jsonl(train_split, train_split_path)
    write_jsonl(validation_split, validation_split_path)
    write_jsonl(test_split, test_split_path)

    report_text = build_report(final_examples, train_split, validation_split, test_split, stats=stats)
    write_text(report_text, report_path)
    logger.info("Saved final dataset to %s", final_dataset_path)
    logger.info("Saved data report to %s", report_path)
    print(report_text)


if __name__ == "__main__":
    main()
