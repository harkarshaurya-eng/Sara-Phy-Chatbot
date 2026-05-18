from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any, Iterable

import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    DataPathsConfig,
    ensure_dir,
    load_train_config,
    normalize_text,
    save_json,
    setup_logging,
    write_jsonl,
    write_text,
)


PHYSICS_KEYWORDS = [
    "physics",
    "mechanics",
    "force",
    "energy",
    "mass",
    "velocity",
    "acceleration",
    "momentum",
    "electric",
    "magnetic",
    "electromagnetic",
    "wave",
    "thermo",
    "temperature",
    "quantum",
    "gauss",
    "newton",
    "entropy",
    "gravity",
    "relativity",
    "optics",
    "photon",
    "electron",
    "nuclear",
    "astrophysics",
]

TOPIC_KEYWORDS: list[tuple[str, list[str]]] = [
    ("mechanics", ["force", "motion", "velocity", "acceleration", "momentum", "newton", "projectile", "gravity"]),
    ("electromagnetism", ["electric", "magnetic", "gauss", "field", "circuit", "charge", "current", "voltage"]),
    ("thermodynamics", ["thermo", "temperature", "heat", "entropy", "engine", "pressure"]),
    ("optics", ["optics", "light", "lens", "mirror", "refraction", "diffraction"]),
    ("waves", ["wave", "frequency", "wavelength", "oscillation", "sound"]),
    ("relativity", ["relativity", "time dilation", "lorentz", "spacetime"]),
    ("quantum physics", ["quantum", "photon", "wavefunction", "uncertainty", "electron"]),
    ("nuclear physics", ["nuclear", "radioactive", "decay", "fission", "fusion"]),
    ("astrophysics", ["star", "planet", "galaxy", "cosmic", "astrophysics", "black hole"]),
]

QUESTION_COLUMNS = {"question", "prompt", "instruction", "query", "input", "user"}
ANSWER_COLUMNS = {"answer", "response", "output", "completion", "assistant", "target"}
TEXT_COLUMNS = {"text", "content", "body", "paragraph", "passage", "document"}
TOPIC_COLUMNS = {"topic", "subject", "category", "domain"}
TITLE_COLUMNS = {"title", "heading", "name"}


@dataclass
class DatasetSpec:
    name: str
    kind: str
    group: str
    enabled: bool = True
    optional: bool = False
    streaming: bool = False
    requires_license_review: bool = False
    dataset_id: str | None = None
    config_name: str | None = None
    config_names: list[str] = field(default_factory=list)
    split_names: list[str] = field(default_factory=lambda: ["train"])
    sample_cap: int | None = None
    license_name: str = "verify-upstream"
    source_url: str = ""
    description: str = ""
    notes: str = ""


DATASET_REGISTRY: list[DatasetSpec] = [
    DatasetSpec(
        name="camel_ai_physics",
        kind="camel_physics",
        group="physics",
        dataset_id="camel-ai/physics",
        config_name="default",
        split_names=["train"],
        sample_cap=20000,
        license_name="cc-by-nc-4.0",
        source_url="https://huggingface.co/datasets/camel-ai/physics",
        description="Physics QA pairs and tutor-style exchanges from CAMEL-AI.",
    ),
    DatasetSpec(
        name="sciq_physics_filtered",
        kind="physics_filtered_qa",
        group="physics",
        dataset_id="allenai/sciq",
        config_name="default",
        split_names=["train", "validation", "test"],
        sample_cap=8000,
        license_name="cc-by-nc-3.0",
        source_url="https://huggingface.co/datasets/allenai/sciq",
        description="General science QA filtered down to physics-heavy samples.",
    ),
    DatasetSpec(
        name="ai2_arc_physics_filtered",
        kind="arc_physics",
        group="physics",
        dataset_id="allenai/ai2_arc",
        config_name="ARC-Challenge",
        split_names=["train", "validation", "test"],
        sample_cap=8000,
        license_name="cc-by-sa-4.0",
        source_url="https://huggingface.co/datasets/allenai/ai2_arc",
        description="AI2 ARC challenge questions filtered to physics-like content.",
    ),
    DatasetSpec(
        name="ai2_arc_easy_physics_filtered",
        kind="arc_physics",
        group="physics",
        dataset_id="allenai/ai2_arc",
        config_name="ARC-Easy",
        split_names=["train", "validation", "test"],
        sample_cap=8000,
        license_name="cc-by-sa-4.0",
        source_url="https://huggingface.co/datasets/allenai/ai2_arc",
        description="AI2 ARC easy questions filtered to physics-like content.",
    ),
    DatasetSpec(
        name="scienceqa_physics_filtered",
        kind="scienceqa_physics",
        group="physics",
        dataset_id="derek-thomas/ScienceQA",
        split_names=["train", "validation", "test"],
        sample_cap=8000,
        license_name="cc-by-nc-sa-4.0",
        source_url="https://huggingface.co/datasets/derek-thomas/ScienceQA",
        description="ScienceQA rows filtered to physics-related content.",
    ),
    DatasetSpec(
        name="ugphysics_english",
        kind="ugphysics",
        group="physics",
        dataset_id="UGPhysics/ugphysics",
        config_names=[
            "AtomicPhysics",
            "ClassicalElectromagnetism",
            "ClassicalMechanics",
            "Electrodynamics",
            "GeometricalOptics",
            "QuantumMechanics",
            "Relativity",
            "SemiconductorPhysics",
            "Solid-StatePhysics",
            "StatisticalMechanics",
            "TheoreticalMechanics",
            "Thermodynamics",
            "WaveOptics",
        ],
        split_names=["en"],
        sample_cap=12000,
        license_name="cc-by-nc-sa-4.0",
        source_url="https://huggingface.co/datasets/UGPhysics/ugphysics",
        description="University-level physics problem solving dataset in English.",
    ),
    DatasetSpec(
        name="mmlu_physics_subsets",
        kind="mmlu_physics",
        group="physics",
        dataset_id="tasksource/mmlu",
        config_names=["conceptual_physics", "college_physics", "high_school_physics"],
        split_names=["test"],
        sample_cap=4000,
        requires_license_review=True,
        license_name="verify-upstream",
        source_url="https://huggingface.co/datasets/tasksource/mmlu",
        description="Physics-related MMLU subsets.",
        notes="Useful as extra conceptual physics text, but verify benchmark reuse terms.",
    ),
    DatasetSpec(
        name="openbookqa_physics_filtered",
        kind="openbookqa",
        group="physics",
        dataset_id="allenai/openbookqa",
        config_name="main",
        split_names=["train", "validation", "test"],
        sample_cap=3000,
        enabled=False,
        requires_license_review=True,
        license_name="verify-upstream",
        source_url="https://huggingface.co/datasets/allenai/openbookqa",
        description="OpenBookQA filtered to physics-like questions.",
        notes="Kept in the registry but disabled by default until you verify reuse terms.",
    ),
    DatasetSpec(
        name="openstax_college_physics",
        kind="openstax_pdf",
        group="open_text",
        optional=True,
        enabled=False,
        sample_cap=4000,
        license_name="cc-by-4.0",
        source_url="https://openstax.org/details/books/college-physics-2e",
        description="Optional OpenStax College Physics PDF ingestion.",
    ),
    DatasetSpec(
        name="oasst1_english_pairs",
        kind="oasst1_pairs",
        group="conversation",
        dataset_id="OpenAssistant/oasst1",
        config_name="default",
        split_names=["train"],
        sample_cap=12000,
        license_name="apache-2.0",
        source_url="https://huggingface.co/datasets/OpenAssistant/oasst1",
        description="English user-assistant pairs reconstructed from OpenAssistant threads.",
    ),
    DatasetSpec(
        name="oasst2_english_pairs",
        kind="oasst2_pairs",
        group="conversation",
        dataset_id="OpenAssistant/oasst2",
        split_names=["train", "validation"],
        sample_cap=12000,
        license_name="apache-2.0",
        source_url="https://huggingface.co/datasets/OpenAssistant/oasst2",
        description="Larger OpenAssistant English conversation pairs.",
    ),
    DatasetSpec(
        name="ultrachat_200k_pairs",
        kind="chat_messages",
        group="conversation",
        dataset_id="HuggingFaceH4/ultrachat_200k",
        split_names=["train_sft"],
        streaming=True,
        sample_cap=12000,
        license_name="mit",
        source_url="https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k",
        description="General chat instruction data sampled into user-assistant pairs.",
    ),
    DatasetSpec(
        name="dolly_15k",
        kind="dolly",
        group="conversation",
        dataset_id="databricks/databricks-dolly-15k",
        split_names=["train"],
        sample_cap=10000,
        license_name="cc-by-sa-3.0",
        source_url="https://huggingface.co/datasets/databricks/databricks-dolly-15k",
        description="Instruction-following and dialogue-style samples from Dolly 15k.",
    ),
    DatasetSpec(
        name="openorca_sampled",
        kind="openorca",
        group="conversation",
        dataset_id="Open-Orca/OpenOrca",
        split_names=["train"],
        streaming=True,
        sample_cap=12000,
        license_name="mit",
        source_url="https://huggingface.co/datasets/Open-Orca/OpenOrca",
        description="Large general conversation and instruction corpus sampled for Colab-friendly use.",
        notes="Huge dataset. Stream and sample it for Colab-sized runs.",
    ),
    DatasetSpec(
        name="slimorca_pairs",
        kind="sharegpt_conversations",
        group="conversation",
        dataset_id="Open-Orca/SlimOrca",
        split_names=["train"],
        streaming=True,
        sample_cap=10000,
        license_name="mit",
        source_url="https://huggingface.co/datasets/Open-Orca/SlimOrca",
        description="Smaller MIT-licensed ShareGPT-style instruction conversations.",
    ),
    DatasetSpec(
        name="dailydialog_pairs",
        kind="daily_dialog",
        group="conversation",
        dataset_id="roskoN/dailydialog",
        split_names=["train", "validation", "test"],
        sample_cap=8000,
        license_name="cc-by-nc-sa-4.0",
        source_url="https://huggingface.co/datasets/roskoN/dailydialog",
        description="Daily multi-turn dialogue pairs for more natural chat behavior.",
    ),
    DatasetSpec(
        name="local_user_files",
        kind="local_files",
        group="local",
        sample_cap=20000,
        license_name="user-provided",
        description="Local TXT, MD, PDF, CSV, TSV, JSON, and JSONL files under data/raw/local/.",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download or normalize public physics and conversation datasets into data/raw.")
    parser.add_argument("--train-config", default="configs/train_config.yaml", help="Training config YAML path.")
    parser.add_argument("--only-source", action="append", help="Download just one or more named sources.")
    parser.add_argument(
        "--only-group",
        action="append",
        choices=["physics", "conversation", "open_text", "local"],
        help="Restrict the run to one or more dataset groups.",
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Include optional web sources such as OpenStax if available.",
    )
    parser.add_argument(
        "--include-review-sources",
        action="store_true",
        help="Include public sources that still require manual license review before training or redistribution.",
    )
    parser.add_argument(
        "--max-samples-per-dataset",
        type=int,
        help="Override the global sample cap from configs/train_config.yaml.",
    )
    parser.add_argument("--list-sources", action="store_true", help="Print the dataset registry and exit.")
    return parser.parse_args()


def load_hf_dataset(dataset_id: str, split: str, config_name: str | None = None, streaming: bool = False):
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"path": dataset_id, "split": split, "streaming": streaming}
    if config_name:
        kwargs["name"] = config_name
    return load_dataset(**kwargs)


def effective_limit(spec: DatasetSpec, global_limit: int) -> int:
    if spec.sample_cap is None:
        return global_limit
    return max(1, min(global_limit, spec.sample_cap))


def format_choices(choices: Any) -> tuple[list[str], str]:
    labels: list[str] = []
    texts: list[str] = []
    if isinstance(choices, dict):
        labels = [normalize_text(label) for label in choices.get("label", [])]
        texts = [normalize_text(text) for text in choices.get("text", [])]
    elif isinstance(choices, list):
        labels = [chr(ord("A") + index) for index in range(len(choices))]
        texts = [normalize_text(choice) for choice in choices]
    rendered = "\n".join(f"{label}. {text}" for label, text in zip(labels, texts) if text)
    return texts, rendered


def resolve_choice_answer(choices: Any, answer_ref: Any) -> str:
    texts, _ = format_choices(choices)
    if not texts:
        return normalize_text(answer_ref)

    if isinstance(choices, dict):
        labels = [normalize_text(label) for label in choices.get("label", [])]
    else:
        labels = [chr(ord("A") + index) for index in range(len(texts))]

    if isinstance(answer_ref, int) and 0 <= answer_ref < len(texts):
        return texts[answer_ref]

    answer_text = normalize_text(answer_ref)
    if answer_text.isdigit():
        index = int(answer_text)
        if 0 <= index < len(texts):
            return texts[index]

    for label, text in zip(labels, texts):
        if answer_text == label or answer_text.lower() == text.lower():
            return text
    return answer_text


def looks_physics_like(text: str) -> bool:
    lowered = normalize_text(text).lower()
    return any(keyword in lowered for keyword in PHYSICS_KEYWORDS)


def infer_topic_from_text(text: str, fallback: str = "general physics") -> str:
    lowered = normalize_text(text).lower()
    for topic, keywords in TOPIC_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return topic
    return fallback


def stringify_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return normalize_text(value)
    if isinstance(value, (int, float, bool)):
        return normalize_text(value)
    if isinstance(value, list):
        parts = [stringify_value(item) for item in value]
        return normalize_text(" | ".join(part for part in parts if part))
    if isinstance(value, dict):
        parts = [f"{normalize_text(key)}: {stringify_value(item)}" for key, item in value.items()]
        return normalize_text(" | ".join(part for part in parts if part))
    return normalize_text(value)


def normalize_record(record: dict[str, Any]) -> dict[str, Any] | None:
    question = normalize_text(record.get("question"))
    answer = normalize_text(record.get("answer"))
    text = normalize_text(record.get("text"))
    if question and answer:
        return {
            "record_type": "qa",
            "source": record["source"],
            "topic": record.get("topic", "general physics"),
            "license": record.get("license", "verify-upstream"),
            "question": question,
            "answer": answer,
            "metadata": record.get("metadata", {}),
        }
    if text:
        return {
            "record_type": "text",
            "source": record["source"],
            "topic": record.get("topic", "general physics"),
            "license": record.get("license", "verify-upstream"),
            "title": normalize_text(record.get("title", record.get("source", "physics notes"))),
            "text": text,
            "metadata": record.get("metadata", {}),
        }
    return None


def extract_pdf_text(file_path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(file_path))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    return "\n\n".join(page for page in pages if page)


def discover_openstax_pdf(book_page_url: str) -> str | None:
    response = requests.get(book_page_url, timeout=60)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        if href.lower().endswith(".pdf"):
            return requests.compat.urljoin(book_page_url, href)
    return None


def iter_hf_rows(spec: DatasetSpec, logger, config_name: str | None = None) -> Iterable[dict[str, Any]]:
    dataset_id = spec.dataset_id or ""
    config_to_use = config_name if config_name is not None else spec.config_name
    for split_name in spec.split_names or ["train"]:
        try:
            dataset = load_hf_dataset(
                dataset_id,
                split=split_name,
                config_name=config_to_use,
                streaming=spec.streaming,
            )
        except Exception as exc:
            logger.warning("Skipping split %s for %s because it failed to load: %s", split_name, spec.name, exc)
            continue
        for row in dataset:
            if isinstance(row, dict):
                yield row


def append_record_if_valid(rows: list[dict[str, Any]], payload: dict[str, Any], max_samples: int) -> bool:
    normalized = normalize_record(payload)
    if normalized is None:
        return False
    rows.append(normalized)
    return len(rows) >= max_samples


def download_camel_physics(spec: DatasetSpec, max_samples: int, logger) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in iter_hf_rows(spec, logger=logger):
        topic = normalize_text(row.get("topic;") or row.get("topic") or row.get("sub_topic")) or "general physics"
        done = append_record_if_valid(
            rows,
            {
                "source": spec.name,
                "topic": topic,
                "license": spec.license_name,
                "question": row.get("message_1"),
                "answer": row.get("message_2"),
                "metadata": {
                    "sub_topic": row.get("sub_topic"),
                    "role_1": row.get("role_1"),
                    "message_id": row.get("message_id"),
                },
            },
            max_samples=max_samples,
        )
        if done:
            break
    logger.info("Collected %s camel-ai/physics rows.", len(rows))
    return rows


def download_sciq(spec: DatasetSpec, max_samples: int, logger) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in iter_hf_rows(spec, logger=logger):
        question = normalize_text(row.get("question"))
        support = normalize_text(row.get("support"))
        answer = normalize_text(row.get("correct_answer"))
        combined = " ".join([question, support, answer])
        if not looks_physics_like(combined):
            continue
        if support:
            answer = f"{answer}\n\nSupport: {support}"
        done = append_record_if_valid(
            rows,
            {
                "source": spec.name,
                "topic": infer_topic_from_text(combined),
                "license": spec.license_name,
                "question": question,
                "answer": answer,
            },
            max_samples=max_samples,
        )
        if done:
            break
    logger.info("Collected %s filtered SciQ rows.", len(rows))
    return rows


def download_arc_physics(spec: DatasetSpec, max_samples: int, logger) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in iter_hf_rows(spec, logger=logger):
        texts, rendered_choices = format_choices(row.get("choices"))
        question = normalize_text(row.get("question"))
        fact = normalize_text(row.get("fact1"))
        answer = resolve_choice_answer(row.get("choices"), row.get("answerKey"))
        combined = " ".join([question, " ".join(texts), fact, answer])
        if not looks_physics_like(combined):
            continue
        prompt = question
        if rendered_choices:
            prompt = f"{question}\nChoices:\n{rendered_choices}"
        if fact:
            answer = f"{answer}\n\nReference fact: {fact}" if answer else fact
        done = append_record_if_valid(
            rows,
            {
                "source": spec.name,
                "topic": infer_topic_from_text(combined),
                "license": spec.license_name,
                "question": prompt,
                "answer": answer,
            },
            max_samples=max_samples,
        )
        if done:
            break
    logger.info("Collected %s filtered ARC rows.", len(rows))
    return rows


def download_openbookqa(spec: DatasetSpec, max_samples: int, logger) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in iter_hf_rows(spec, logger=logger):
        texts, rendered_choices = format_choices(row.get("choices"))
        question = normalize_text(row.get("question_stem"))
        fact = normalize_text(row.get("fact1"))
        answer = resolve_choice_answer(row.get("choices"), row.get("answerKey"))
        combined = " ".join([question, " ".join(texts), fact, answer])
        if not looks_physics_like(combined):
            continue
        prompt = question
        if rendered_choices:
            prompt = f"{question}\nChoices:\n{rendered_choices}"
        if fact:
            answer = f"{answer}\n\nKnown fact: {fact}" if answer else fact
        done = append_record_if_valid(
            rows,
            {
                "source": spec.name,
                "topic": infer_topic_from_text(combined),
                "license": spec.license_name,
                "question": prompt,
                "answer": answer,
            },
            max_samples=max_samples,
        )
        if done:
            break
    logger.info("Collected %s filtered OpenBookQA rows.", len(rows))
    return rows


def download_mmlu_physics(spec: DatasetSpec, max_samples: int, logger) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    subset_names = spec.config_names or []
    per_subset_cap = max(1, max_samples // max(1, len(subset_names)))
    for config_name in subset_names:
        subset_rows = 0
        for row in iter_hf_rows(spec, logger=logger, config_name=config_name):
            texts, rendered_choices = format_choices(row.get("choices"))
            question = normalize_text(row.get("question"))
            answer = resolve_choice_answer(row.get("choices"), row.get("answer"))
            prompt = question
            if rendered_choices:
                prompt = f"{question}\nChoices:\n{rendered_choices}"
            done = append_record_if_valid(
                rows,
                {
                    "source": spec.name,
                    "topic": config_name.replace("_", " "),
                    "license": spec.license_name,
                    "question": prompt,
                    "answer": answer,
                    "metadata": {"subset": config_name},
                },
                max_samples=max_samples,
            )
            if done:
                break
            subset_rows += 1
            if subset_rows >= per_subset_cap:
                break
        logger.info("Collected %s MMLU rows from subset %s.", subset_rows, config_name)
        if len(rows) >= max_samples:
            break
    return rows


def download_scienceqa_physics(spec: DatasetSpec, max_samples: int, logger) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in iter_hf_rows(spec, logger=logger):
        subject = normalize_text(row.get("subject")).lower()
        question = normalize_text(row.get("question"))
        hint = normalize_text(row.get("hint"))
        lecture = normalize_text(row.get("lecture"))
        solution = normalize_text(row.get("solution"))
        choices = row.get("choices") or []
        answer_index = row.get("answer")
        answer = ""
        if isinstance(answer_index, int) and 0 <= answer_index < len(choices):
            answer = normalize_text(choices[answer_index])
        combined = " ".join([subject, question, hint, lecture, solution, stringify_value(choices), answer])
        if subject and "physics" not in subject and not looks_physics_like(combined):
            continue

        rendered_choices = "\n".join(
            f"{chr(ord('A') + index)}. {normalize_text(choice)}"
            for index, choice in enumerate(choices)
            if normalize_text(choice)
        )
        prompt_parts = [question]
        if hint:
            prompt_parts.append(f"Hint: {hint}")
        if rendered_choices:
            prompt_parts.append(f"Choices:\n{rendered_choices}")
        answer_parts = [part for part in [answer, lecture, solution] if part]
        done = append_record_if_valid(
            rows,
            {
                "source": spec.name,
                "topic": infer_topic_from_text(combined),
                "license": spec.license_name,
                "question": "\n\n".join(part for part in prompt_parts if part),
                "answer": "\n\n".join(answer_parts),
                "metadata": {
                    "grade": normalize_text(row.get("grade")),
                    "subject": subject,
                    "topic": normalize_text(row.get("topic")),
                },
            },
            max_samples=max_samples,
        )
        if done:
            break
    logger.info("Collected %s ScienceQA physics rows.", len(rows))
    return rows


def download_ugphysics(spec: DatasetSpec, max_samples: int, logger) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    config_names = spec.config_names or []
    per_subset_cap = max(1, max_samples // max(1, len(config_names)))
    for config_name in config_names:
        subset_rows = 0
        for row in iter_hf_rows(spec, logger=logger, config_name=config_name):
            question = normalize_text(row.get("problem") or row.get("question") or row.get("input"))
            solution = stringify_value(row.get("solution") or row.get("analysis") or row.get("rationale"))
            answer = stringify_value(row.get("answer") or row.get("answers") or row.get("final_answer"))
            if not question:
                continue
            answer_text = solution or answer
            if solution and answer and answer.lower() not in solution.lower():
                answer_text = f"{solution}\n\nFinal answer: {answer}"
            if not answer_text:
                continue
            topic = normalize_text(row.get("topic") or row.get("subject")) or config_name.replace("_", " ")
            done = append_record_if_valid(
                rows,
                {
                    "source": spec.name,
                    "topic": infer_topic_from_text(f"{question}\n{answer_text}", fallback=topic),
                    "license": spec.license_name,
                    "question": question,
                    "answer": answer_text,
                    "metadata": {
                        "subset": config_name,
                        "level": normalize_text(row.get("level")),
                        "unit": normalize_text(row.get("unit")),
                    },
                },
                max_samples=max_samples,
            )
            subset_rows += 1
            if done or subset_rows >= per_subset_cap:
                break
        logger.info("Collected %s UGPhysics rows from subset %s.", subset_rows, config_name)
        if len(rows) >= max_samples:
            break
    return rows


def conversation_messages_to_records(
    messages: Any,
    source: str,
    license_name: str,
    default_topic: str = "general conversation",
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        return []
    rows: list[dict[str, Any]] = []
    previous_user = ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = normalize_text(message.get("role") or message.get("from") or message.get("speaker")).lower()
        if role in {"human", "prompter"}:
            role = "user"
        elif role in {"gpt", "assistant", "bot"}:
            role = "assistant"
        content = normalize_text(message.get("content") or message.get("text") or message.get("value") or message.get("utterance"))
        if not content:
            continue
        if role == "user":
            previous_user = content
            continue
        if role == "assistant" and previous_user:
            topic = infer_topic_from_text(f"{previous_user}\n{content}", fallback=default_topic)
            normalized = normalize_record(
                {
                    "source": source,
                    "topic": topic,
                    "license": license_name,
                    "question": previous_user,
                    "answer": content,
                    "metadata": metadata or {},
                }
            )
            if normalized is not None:
                rows.append(normalized)
            previous_user = ""
    return rows


def download_ultrachat(spec: DatasetSpec, max_samples: int, logger) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in iter_hf_rows(spec, logger=logger):
        records = conversation_messages_to_records(
            row.get("messages"),
            source=spec.name,
            license_name=spec.license_name,
            metadata={"prompt": normalize_text(row.get("prompt"))},
        )
        for record in records:
            rows.append(record)
            if len(rows) >= max_samples:
                logger.info("Collected %s UltraChat rows.", len(rows))
                return rows
    logger.info("Collected %s UltraChat rows.", len(rows))
    return rows


def download_dolly(spec: DatasetSpec, max_samples: int, logger) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in iter_hf_rows(spec, logger=logger):
        instruction = normalize_text(row.get("instruction"))
        context = normalize_text(row.get("context"))
        answer = normalize_text(row.get("response"))
        if not instruction or not answer:
            continue
        question = instruction if not context else f"{instruction}\n\nContext:\n{context}"
        topic = infer_topic_from_text(
            f"{instruction}\n{context}\n{answer}",
            fallback=normalize_text(row.get("category")) or "general conversation",
        )
        done = append_record_if_valid(
            rows,
            {
                "source": spec.name,
                "topic": topic,
                "license": spec.license_name,
                "question": question,
                "answer": answer,
                "metadata": {"category": normalize_text(row.get("category"))},
            },
            max_samples=max_samples,
        )
        if done:
            break
    logger.info("Collected %s Dolly rows.", len(rows))
    return rows


def download_openorca(spec: DatasetSpec, max_samples: int, logger) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in iter_hf_rows(spec, logger=logger):
        question = normalize_text(row.get("question") or row.get("prompt"))
        answer = normalize_text(row.get("response") or row.get("output"))
        if not question or not answer:
            continue
        system_prompt = normalize_text(row.get("system_prompt"))
        metadata = {}
        if system_prompt:
            metadata["system_prompt"] = system_prompt
        done = append_record_if_valid(
            rows,
            {
                "source": spec.name,
                "topic": infer_topic_from_text(f"{question}\n{answer}", fallback="general conversation"),
                "license": spec.license_name,
                "question": question,
                "answer": answer,
                "metadata": metadata,
            },
            max_samples=max_samples,
        )
        if done:
            break
    logger.info("Collected %s OpenOrca rows.", len(rows))
    return rows


def download_oasst_pairs(spec: DatasetSpec, max_samples: int, logger) -> list[dict[str, Any]]:
    dataset_rows = list(iter_hf_rows(spec, logger=logger))
    by_id: dict[str, dict[str, Any]] = {}
    ordered_rows: list[dict[str, Any]] = []
    for row in dataset_rows:
        message_id = normalize_text(row.get("message_id"))
        language = normalize_text(row.get("lang")).lower()
        if language and not language.startswith("en"):
            continue
        if message_id:
            by_id[message_id] = row
            ordered_rows.append(row)

    rows: list[dict[str, Any]] = []
    for row in ordered_rows:
        role = normalize_text(row.get("role")).lower()
        if role != "assistant":
            continue
        parent_id = normalize_text(row.get("parent_id"))
        parent = by_id.get(parent_id)
        if not parent:
            continue
        if normalize_text(parent.get("role")).lower() != "prompter":
            continue
        question = normalize_text(parent.get("text"))
        answer = normalize_text(row.get("text"))
        if not question or not answer:
            continue
        done = append_record_if_valid(
            rows,
            {
                "source": spec.name,
                "topic": infer_topic_from_text(f"{question}\n{answer}", fallback="general conversation"),
                "license": spec.license_name,
                "question": question,
                "answer": answer,
                "metadata": {
                    "assistant_message_id": normalize_text(row.get("message_id")),
                    "user_message_id": normalize_text(parent.get("message_id")),
                },
            },
            max_samples=max_samples,
        )
        if done:
            break
    logger.info("Collected %s OpenAssistant pair rows.", len(rows))
    return rows


def download_sharegpt_conversations(spec: DatasetSpec, max_samples: int, logger) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in iter_hf_rows(spec, logger=logger):
        records = conversation_messages_to_records(
            row.get("conversations") or row.get("messages"),
            source=spec.name,
            license_name=spec.license_name,
            metadata={"system_prompt": normalize_text(row.get("system_prompt"))},
        )
        for record in records:
            rows.append(record)
            if len(rows) >= max_samples:
                logger.info("Collected %s ShareGPT-style conversation rows.", len(rows))
                return rows
    logger.info("Collected %s ShareGPT-style conversation rows.", len(rows))
    return rows


def download_dailydialog(spec: DatasetSpec, max_samples: int, logger) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in iter_hf_rows(spec, logger=logger):
        dialogue = row.get("dialog") or row.get("utterances") or row.get("dialogue")
        if not isinstance(dialogue, list):
            continue
        normalized_turns = [normalize_text(turn) for turn in dialogue if normalize_text(turn)]
        for index in range(0, len(normalized_turns) - 1, 2):
            question = normalized_turns[index]
            answer = normalized_turns[index + 1]
            done = append_record_if_valid(
                rows,
                {
                    "source": spec.name,
                    "topic": infer_topic_from_text(f"{question}\n{answer}", fallback="general conversation"),
                    "license": spec.license_name,
                    "question": question,
                    "answer": answer,
                },
                max_samples=max_samples,
            )
            if done:
                logger.info("Collected %s DailyDialog rows.", len(rows))
                return rows
    logger.info("Collected %s DailyDialog rows.", len(rows))
    return rows


def download_openstax(spec: DatasetSpec, raw_dir: Path, logger) -> list[dict[str, Any]]:
    pdf_url = discover_openstax_pdf(spec.source_url)
    if not pdf_url:
        raise RuntimeError("Could not discover an OpenStax PDF link from the book page.")
    assets_dir = ensure_dir(raw_dir / "assets")
    pdf_path = assets_dir / "openstax_college_physics.pdf"
    response = requests.get(pdf_url, timeout=120)
    response.raise_for_status()
    pdf_path.write_bytes(response.content)
    text = extract_pdf_text(pdf_path)
    return [
        {
            "record_type": "text",
            "source": spec.name,
            "topic": "general physics",
            "license": spec.license_name,
            "title": "OpenStax College Physics 2e",
            "text": text,
            "metadata": {"pdf_url": pdf_url},
        }
    ]


def normalized_mapping_keys(mapping: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in mapping.items():
        key_text = normalize_text(key).lower()
        if key_text:
            normalized[key_text] = value
    return normalized


def pick_first_value(mapping: dict[str, Any], keys: set[str]) -> str:
    for key in keys:
        value = normalize_text(mapping.get(key))
        if value:
            return value
    return ""


def local_row_to_records(file_path: Path, row: dict[str, Any]) -> list[dict[str, Any]]:
    normalized = normalized_mapping_keys(row)
    topic = pick_first_value(normalized, TOPIC_COLUMNS) or "general physics"
    if "messages" in normalized:
        try:
            messages = normalized["messages"]
            if isinstance(messages, str):
                messages = json.loads(messages)
        except Exception:
            messages = normalized["messages"]
        return conversation_messages_to_records(
            messages,
            source=f"local::{file_path.stem}",
            license_name="user-provided",
            default_topic=topic,
        )

    question = pick_first_value(normalized, QUESTION_COLUMNS)
    answer = pick_first_value(normalized, ANSWER_COLUMNS)
    if question and answer:
        record = normalize_record(
            {
                "source": f"local::{file_path.stem}",
                "topic": infer_topic_from_text(f"{question}\n{answer}", fallback=topic),
                "license": "user-provided",
                "question": question,
                "answer": answer,
                "metadata": {"file_name": file_path.name},
            }
        )
        return [record] if record is not None else []

    text = pick_first_value(normalized, TEXT_COLUMNS)
    if text:
        title = pick_first_value(normalized, TITLE_COLUMNS) or file_path.stem
        record = normalize_record(
            {
                "source": f"local::{file_path.stem}",
                "topic": infer_topic_from_text(text, fallback=topic),
                "license": "user-provided",
                "title": title,
                "text": text,
                "metadata": {"file_name": file_path.name},
            }
        )
        return [record] if record is not None else []
    return []


def read_local_json_file(file_path: Path, logger) -> list[dict[str, Any]]:
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Skipping JSON file %s because it could not be parsed: %s", file_path, exc)
        return []
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


def read_local_jsonl_file(file_path: Path, logger) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning("Skipping bad JSONL row in %s line %s: %s", file_path, line_number, exc)
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def read_local_table_file(file_path: Path, delimiter: str, logger) -> list[dict[str, Any]]:
    try:
        with file_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            if not reader.fieldnames:
                logger.warning("Skipping %s because it has no header row.", file_path)
                return []
            return [dict(row) for row in reader]
    except Exception as exc:
        logger.warning("Skipping tabular file %s because it could not be read: %s", file_path, exc)
        return []


def download_local_files(data_paths: DataPathsConfig, max_samples: int, logger) -> list[dict[str, Any]]:
    local_dir = PROJECT_ROOT / data_paths.local_text_dir
    ensure_dir(local_dir)
    rows: list[dict[str, Any]] = []

    for file_path in sorted(local_dir.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.name.startswith(".") or file_path.name.lower().startswith("readme"):
            continue
        suffix = file_path.suffix.lower()
        file_records: list[dict[str, Any]] = []
        try:
            if suffix in {".txt", ".md"}:
                text = file_path.read_text(encoding="utf-8")
                record = normalize_record(
                    {
                        "source": f"local::{file_path.stem}",
                        "topic": infer_topic_from_text(text, fallback="general physics"),
                        "license": "user-provided",
                        "title": file_path.stem,
                        "text": text,
                        "metadata": {"file_name": file_path.name},
                    }
                )
                file_records = [record] if record is not None else []
            elif suffix == ".pdf":
                text = extract_pdf_text(file_path)
                record = normalize_record(
                    {
                        "source": f"local::{file_path.stem}",
                        "topic": infer_topic_from_text(text, fallback="general physics"),
                        "license": "user-provided",
                        "title": file_path.stem,
                        "text": text,
                        "metadata": {"file_name": file_path.name},
                    }
                )
                file_records = [record] if record is not None else []
            elif suffix == ".csv":
                for row in read_local_table_file(file_path, delimiter=",", logger=logger):
                    file_records.extend(local_row_to_records(file_path, row))
            elif suffix == ".tsv":
                for row in read_local_table_file(file_path, delimiter="\t", logger=logger):
                    file_records.extend(local_row_to_records(file_path, row))
            elif suffix == ".json":
                for row in read_local_json_file(file_path, logger=logger):
                    file_records.extend(local_row_to_records(file_path, row))
            elif suffix == ".jsonl":
                for row in read_local_jsonl_file(file_path, logger=logger):
                    file_records.extend(local_row_to_records(file_path, row))
            else:
                continue
        except Exception as exc:  # pragma: no cover - file dependent
            logger.warning("Skipping local file %s because it could not be processed: %s", file_path, exc)
            continue

        for record in file_records:
            rows.append(record)
            if len(rows) >= max_samples:
                logger.info("Collected %s local file records.", len(rows))
                return rows
    logger.info("Collected %s local file records.", len(rows))
    return rows


def build_sample_records() -> list[dict[str, Any]]:
    return [
        {
            "record_type": "qa",
            "source": "sample_seed",
            "topic": "mechanics",
            "license": "sample",
            "question": "Explain Newton's second law.",
            "answer": "Newton's second law says that the net force on an object equals its mass times its acceleration: F = ma.",
        },
        {
            "record_type": "qa",
            "source": "sample_seed",
            "topic": "electromagnetism",
            "license": "sample",
            "question": "What is Gauss's law?",
            "answer": "Gauss's law states that the electric flux through a closed surface equals the enclosed charge divided by the permittivity of free space.",
        },
        {
            "record_type": "qa",
            "source": "sample_seed",
            "topic": "general conversation",
            "license": "sample",
            "question": "Can you explain entropy simply?",
            "answer": "Entropy is a measure of how spread out energy is or how many microscopic arrangements match the same visible state.",
        },
    ]


def list_sources() -> list[dict[str, Any]]:
    return [
        {
            "name": spec.name,
            "kind": spec.kind,
            "group": spec.group,
            "enabled": spec.enabled,
            "optional": spec.optional,
            "streaming": spec.streaming,
            "requires_license_review": spec.requires_license_review,
            "dataset_id": spec.dataset_id,
            "config_name": spec.config_name,
            "config_names": spec.config_names,
            "split_names": spec.split_names,
            "sample_cap": spec.sample_cap,
            "license": spec.license_name,
            "url": spec.source_url,
            "description": spec.description,
            "notes": spec.notes,
        }
        for spec in DATASET_REGISTRY
    ]


def run_source(spec: DatasetSpec, data_paths: DataPathsConfig, max_samples: int, raw_dir: Path, logger) -> list[dict[str, Any]]:
    if spec.kind == "camel_physics":
        return download_camel_physics(spec, max_samples=max_samples, logger=logger)
    if spec.kind == "physics_filtered_qa":
        return download_sciq(spec, max_samples=max_samples, logger=logger)
    if spec.kind == "arc_physics":
        return download_arc_physics(spec, max_samples=max_samples, logger=logger)
    if spec.kind == "scienceqa_physics":
        return download_scienceqa_physics(spec, max_samples=max_samples, logger=logger)
    if spec.kind == "ugphysics":
        return download_ugphysics(spec, max_samples=max_samples, logger=logger)
    if spec.kind == "openbookqa":
        return download_openbookqa(spec, max_samples=max_samples, logger=logger)
    if spec.kind == "mmlu_physics":
        return download_mmlu_physics(spec, max_samples=max_samples, logger=logger)
    if spec.kind == "chat_messages":
        return download_ultrachat(spec, max_samples=max_samples, logger=logger)
    if spec.kind == "dolly":
        return download_dolly(spec, max_samples=max_samples, logger=logger)
    if spec.kind == "openorca":
        return download_openorca(spec, max_samples=max_samples, logger=logger)
    if spec.kind in {"oasst1_pairs", "oasst2_pairs"}:
        return download_oasst_pairs(spec, max_samples=max_samples, logger=logger)
    if spec.kind == "sharegpt_conversations":
        return download_sharegpt_conversations(spec, max_samples=max_samples, logger=logger)
    if spec.kind == "daily_dialog":
        return download_dailydialog(spec, max_samples=max_samples, logger=logger)
    if spec.kind == "openstax_pdf":
        return download_openstax(spec, raw_dir=raw_dir, logger=logger)
    if spec.kind == "local_files":
        return download_local_files(data_paths=data_paths, max_samples=max_samples, logger=logger)
    raise ValueError(f"Unsupported dataset kind: {spec.kind}")


def json_dumps(payload: Any) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    train_cfg = load_train_config(PROJECT_ROOT, args.train_config)
    log_dir = ensure_dir(PROJECT_ROOT / "logs")
    logger = setup_logging("download_datasets", log_file=log_dir / "download_datasets.log")

    if args.list_sources:
        print(json_dumps(list_sources()))
        return

    data_paths = train_cfg.data
    raw_dir = ensure_dir(PROJECT_ROOT / data_paths.raw_dir)
    ensure_dir(PROJECT_ROOT / data_paths.sample_dir)

    selected_sources = set(args.only_source or [])
    selected_groups = set(args.only_group or [])
    global_max_samples = max(1, int(args.max_samples_per_dataset or data_paths.max_samples_per_dataset))

    manifest: dict[str, Any] = {
        "sources": [],
        "successful_sources": 0,
        "failed_sources": 0,
        "skipped_sources": 0,
        "max_samples_per_dataset": global_max_samples,
    }
    successful_records = 0

    for spec in DATASET_REGISTRY:
        if selected_sources and spec.name not in selected_sources:
            continue
        if selected_groups and spec.group not in selected_groups:
            continue
        if spec.optional and not args.include_optional:
            manifest["sources"].append({"name": spec.name, "status": "skipped", "reason": "optional source not enabled"})
            manifest["skipped_sources"] += 1
            continue
        if spec.requires_license_review and not args.include_review_sources:
            manifest["sources"].append(
                {
                    "name": spec.name,
                    "status": "skipped",
                    "reason": "requires manual license review",
                    "license": spec.license_name,
                    "url": spec.source_url,
                }
            )
            manifest["skipped_sources"] += 1
            continue
        if not spec.enabled:
            manifest["sources"].append({"name": spec.name, "status": "disabled"})
            manifest["skipped_sources"] += 1
            continue

        source_limit = effective_limit(spec, global_max_samples)
        try:
            rows = run_source(spec, data_paths=data_paths, max_samples=source_limit, raw_dir=raw_dir, logger=logger)
            output_path = raw_dir / f"{spec.name}.jsonl"
            write_jsonl(rows, output_path)
            successful_records += len(rows)
            manifest["sources"].append(
                {
                    "name": spec.name,
                    "group": spec.group,
                    "status": "ok",
                    "records": len(rows),
                    "cap_used": source_limit,
                    "license": spec.license_name,
                    "path": str(output_path),
                    "url": spec.source_url,
                    "streaming": spec.streaming,
                }
            )
            manifest["successful_sources"] += 1
        except Exception as exc:  # pragma: no cover - network and dataset dependent
            logger.exception("Source %s failed", spec.name)
            manifest["sources"].append(
                {
                    "name": spec.name,
                    "group": spec.group,
                    "status": "failed",
                    "license": spec.license_name,
                    "url": spec.source_url,
                    "reason": str(exc),
                }
            )
            manifest["failed_sources"] += 1

    sample_records = build_sample_records()
    write_jsonl(sample_records, PROJECT_ROOT / data_paths.sample_records_file)
    write_text(
        "\n".join(record["question"] + "\n" + record["answer"] for record in sample_records),
        PROJECT_ROOT / data_paths.sample_corpus_file,
    )

    if successful_records == 0:
        fallback_path = raw_dir / "sample_seed.jsonl"
        write_jsonl(sample_records, fallback_path)
        manifest["sources"].append(
            {
                "name": "sample_seed",
                "status": "fallback-written",
                "records": len(sample_records),
                "path": str(fallback_path),
            }
        )
        logger.warning("All dataset downloads failed or returned zero rows. Wrote fallback sample data to %s", fallback_path)

    save_json(manifest, PROJECT_ROOT / data_paths.manifest_file)
    logger.info("Dataset download manifest written to %s", PROJECT_ROOT / data_paths.manifest_file)
    print(json_dumps(manifest))


if __name__ == "__main__":
    main()
