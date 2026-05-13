from __future__ import annotations

import glob
import math
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from src.train_utils import ensure_dir, save_json, write_jsonl


PHYSICS_FILTER_KEYWORDS = [
    "physics",
    "force",
    "energy",
    "mass",
    "velocity",
    "acceleration",
    "momentum",
    "gravity",
    "electric",
    "magnetic",
    "wave",
    "quantum",
    "thermo",
    "optics",
    "relativity",
    "nuclear",
    "astrophysics",
]

DEFAULT_ROLE_MAP = {
    "human": "user",
    "user": "user",
    "prompter": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "model": "assistant",
    "bot": "assistant",
    "system": "system",
}


@dataclass
class DatasetSource:
    name: str
    type: str
    enabled: bool
    url: str
    license: str
    citation: str
    dataset_id: str | None = None
    config_name: str | None = None
    config_names: list[str] | None = None
    split: str | None = None
    streaming: bool = True
    max_samples: int | None = None
    question_fields: list[str] | None = None
    answer_fields: list[str] | None = None
    support_fields: list[str] | None = None
    system_prompt_fields: list[str] | None = None
    context_fields: list[str] | None = None
    title_fields: list[str] | None = None
    difficulty_field: str | None = None
    filter_keywords: list[str] | None = None
    topic_hint: str | None = None
    query: str | None = None
    require_explicit_license: bool = False
    allowed_licenses: list[str] | None = None
    required_field_values: dict[str, list[str]] | None = None
    path_glob: str | None = None
    text_field: str | None = None
    message_field: str | None = None
    message_role_key: str | None = None
    message_content_key: str | None = None
    message_role_map: dict[str, str] | None = None
    choice_field: str | None = None
    answer_index_field: str | None = None
    language_field: str | None = None
    allowed_languages: list[str] | None = None
    max_messages_per_example: int | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetSource":
        return cls(**payload)

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "enabled": self.enabled,
            "url": self.url,
            "license": self.license,
            "citation": self.citation,
            "dataset_id": self.dataset_id,
            "config_name": self.config_name,
            "config_names": self.config_names,
            "split": self.split,
            "streaming": self.streaming,
            "max_samples": self.max_samples,
            "topic_hint": self.topic_hint,
        }


def load_sources_from_config(config: dict[str, Any]) -> list[DatasetSource]:
    return [DatasetSource.from_dict(source) for source in config.get("datasets", [])]


def _is_known_license(license_name: str) -> bool:
    normalized = (license_name or "").strip().lower()
    return normalized not in {"", "unknown", "unverified", "n/a", "none"}


def _pick_first(example: dict[str, Any], candidates: list[str] | None) -> Any:
    if not candidates:
        return None
    for candidate in candidates:
        value = example.get(candidate)
        if value not in (None, "", []):
            return value
    return None


def _pick_many(example: dict[str, Any], candidates: list[str] | None) -> list[tuple[str, Any]]:
    values: list[tuple[str, Any]] = []
    if not candidates:
        return values
    for candidate in candidates:
        value = example.get(candidate)
        if value not in (None, "", []):
            values.append((candidate, value))
    return values


def _normalize_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(_normalize_scalar(item) for item in value if _normalize_scalar(item))
    if isinstance(value, dict):
        if "text" in value:
            return _normalize_scalar(value["text"])
        return " ".join(_normalize_scalar(item) for item in value.values() if _normalize_scalar(item))
    return str(value).strip()


def _physics_filtered(text: str, keywords: list[str] | None = None) -> bool:
    bag = (text or "").lower()
    active_keywords = keywords or PHYSICS_FILTER_KEYWORDS
    return any(keyword.lower() in bag for keyword in active_keywords)


def _field_value_matches(raw_value: Any, allowed_values: list[str]) -> bool:
    allowed_normalized = {value.strip().lower() for value in allowed_values}
    if isinstance(raw_value, list):
        return any(_field_value_matches(item, allowed_values) for item in raw_value)
    normalized = _normalize_scalar(raw_value).strip().lower()
    return normalized in allowed_normalized


def _record_matches_filters(source: DatasetSource, example: dict[str, Any]) -> bool:
    if source.required_field_values:
        for field_name, allowed_values in source.required_field_values.items():
            if not _field_value_matches(example.get(field_name), allowed_values):
                return False

    if source.allowed_languages:
        language_field = source.language_field or "lang"
        if not _field_value_matches(example.get(language_field), source.allowed_languages):
            return False

    return True


def _normalize_role(role: str, source: DatasetSource) -> str:
    role_map = {**DEFAULT_ROLE_MAP, **(source.message_role_map or {})}
    return role_map.get(role.strip().lower(), role.strip().lower())


def _normalize_message_list(raw_messages: Any, source: DatasetSource) -> list[dict[str, str]]:
    if not isinstance(raw_messages, list):
        return []

    role_key = source.message_role_key or "role"
    content_key = source.message_content_key or "content"
    normalized: list[dict[str, str]] = []

    for raw_message in raw_messages:
        if not isinstance(raw_message, dict):
            continue
        role = _normalize_role(str(raw_message.get(role_key, "")), source)
        content = _normalize_scalar(raw_message.get(content_key))
        if not role or not content:
            continue
        if normalized and normalized[-1]["role"] == role:
            normalized[-1]["content"] = f"{normalized[-1]['content']}\n\n{content}"
        else:
            normalized.append({"role": role, "content": content})

    if source.max_messages_per_example and len(normalized) > source.max_messages_per_example:
        normalized = normalized[-source.max_messages_per_example :]
        while normalized and normalized[0]["role"] == "assistant":
            normalized = normalized[1:]

    return normalized


def _format_choices(raw_choices: Any) -> str:
    if not isinstance(raw_choices, list) or not raw_choices:
        return ""
    rendered = []
    for index, choice in enumerate(raw_choices):
        letter = chr(ord("A") + index)
        rendered.append(f"{letter}. {_normalize_scalar(choice)}")
    return "\n".join(rendered)


def _compose_question_text(source: DatasetSource, example: dict[str, Any]) -> str:
    question = _normalize_scalar(_pick_first(example, source.question_fields))
    if not question:
        return ""

    parts = [question]
    context_values = _pick_many(example, source.context_fields)
    if context_values:
        context_text = "\n\n".join(_normalize_scalar(value) for _, value in context_values if _normalize_scalar(value))
        if context_text:
            parts.append(f"Context:\n{context_text}")

    if source.choice_field:
        choices_text = _format_choices(example.get(source.choice_field))
        if choices_text:
            parts.append(f"Choices:\n{choices_text}")

    return "\n\n".join(part for part in parts if part)


def _compose_answer_text(source: DatasetSource, example: dict[str, Any]) -> str:
    answer = _normalize_scalar(_pick_first(example, source.answer_fields))
    if not answer and source.answer_index_field and source.choice_field:
        raw_index = example.get(source.answer_index_field)
        raw_choices = example.get(source.choice_field)
        if isinstance(raw_index, int) and isinstance(raw_choices, list) and 0 <= raw_index < len(raw_choices):
            answer = _normalize_scalar(raw_choices[raw_index])

    support_values = _pick_many(example, source.support_fields)
    support_sections = []
    for field_name, value in support_values:
        normalized_value = _normalize_scalar(value)
        if not normalized_value:
            continue
        label = field_name.replace("_", " ").strip().title()
        support_sections.append(f"{label}:\n{normalized_value}")

    if support_sections:
        combined_support = "\n\n".join(support_sections)
        answer = f"{answer}\n\n{combined_support}" if answer else combined_support

    return answer


def _build_standard_conversation(
    source: DatasetSource,
    example: dict[str, Any],
    row_id: int,
    messages: list[dict[str, str]],
    config_name: str | None = None,
) -> dict[str, Any] | None:
    assistant_messages = [message for message in messages if message.get("role") == "assistant"]
    if len(messages) < 2 or not assistant_messages:
        return None

    return {
        "id": str(example.get("id", f"{source.name}-{config_name or 'default'}-{row_id}")),
        "record_type": "conversation",
        "source": source.name,
        "license": source.license,
        "citation": source.citation,
        "url": source.url,
        "messages": messages,
        "topic": example.get("topic") or source.topic_hint,
        "difficulty": example.get(source.difficulty_field or "difficulty") or example.get("level"),
        "metadata": {
            "dataset_id": source.dataset_id,
            "config_name": config_name,
        },
    }


def _standardize_hf_example(
    source: DatasetSource,
    example: dict[str, Any],
    row_id: int,
    config_name: str | None = None,
) -> dict[str, Any] | None:
    if not _record_matches_filters(source, example):
        return None

    message_field = source.message_field or ("messages" if isinstance(example.get("messages"), list) else None)
    if message_field:
        normalized_messages = _normalize_message_list(example.get(message_field), source)
        return _build_standard_conversation(source, example, row_id, normalized_messages, config_name=config_name)

    question_text = _compose_question_text(source, example)
    answer_text = _compose_answer_text(source, example)
    system_prompt = _normalize_scalar(_pick_first(example, source.system_prompt_fields))

    if question_text and answer_text and (system_prompt or source.system_prompt_fields):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question_text})
        messages.append({"role": "assistant", "content": answer_text})
        return _build_standard_conversation(source, example, row_id, messages, config_name=config_name)

    if question_text and answer_text:
        combined_text = "\n".join(
            part
            for part in [
                question_text,
                answer_text,
                _normalize_scalar(_pick_first(example, source.title_fields)),
            ]
            if part
        )
        if source.filter_keywords and not _physics_filtered(combined_text, source.filter_keywords):
            return None

        return {
            "id": str(example.get("id", f"{source.name}-{config_name or 'default'}-{row_id}")),
            "record_type": "qa",
            "source": source.name,
            "license": source.license,
            "citation": source.citation,
            "url": source.url,
            "question": question_text,
            "answer": answer_text,
            "title": _normalize_scalar(_pick_first(example, source.title_fields)),
            "topic": example.get("topic") or source.topic_hint,
            "difficulty": example.get(source.difficulty_field or "difficulty") or example.get("level"),
            "metadata": {
                "dataset_id": source.dataset_id,
                "config_name": config_name,
            },
        }

    text_field = source.text_field or ("text" if isinstance(example.get("text"), str) else None)
    if text_field:
        text_value = _normalize_scalar(example.get(text_field))
        if not text_value:
            return None
        combined_text = "\n".join(
            part
            for part in [
                text_value,
                _normalize_scalar(_pick_first(example, source.title_fields)),
            ]
            if part
        )
        if source.filter_keywords and not _physics_filtered(combined_text, source.filter_keywords):
            return None
        return {
            "id": str(example.get("id", f"{source.name}-{config_name or 'default'}-{row_id}")),
            "record_type": "text",
            "source": source.name,
            "license": source.license,
            "citation": source.citation,
            "url": source.url,
            "title": _normalize_scalar(_pick_first(example, source.title_fields)),
            "text": text_value,
            "topic": example.get("topic") or source.topic_hint,
            "difficulty": example.get(source.difficulty_field or "difficulty") or example.get("level"),
            "metadata": {
                "dataset_id": source.dataset_id,
                "config_name": config_name,
            },
        }

    return None


def _download_hf_dataset(
    source: DatasetSource,
    output_path: Path,
    max_samples: int,
    logger,
) -> tuple[int, str | None]:
    from datasets import load_dataset

    config_names = source.config_names or [source.config_name]
    sample_cap = int(source.max_samples or max_samples)
    per_config_cap = max(1, math.ceil(sample_cap / max(1, len(config_names))))

    rows: list[dict[str, Any]] = []
    for config_name in config_names:
        dataset = load_dataset(
            path=source.dataset_id,
            name=config_name,
            split=source.split or "train",
            streaming=source.streaming,
        )
        iterator: Iterable[dict[str, Any]] = dataset if source.streaming else iter(dataset)
        config_rows = 0
        for row_id, example in enumerate(iterator):
            standardized = _standardize_hf_example(source, dict(example), row_id=row_id, config_name=config_name)
            if standardized is None:
                continue
            rows.append(standardized)
            config_rows += 1
            if config_rows >= per_config_cap or len(rows) >= sample_cap:
                break
        logger.info(
            "Collected %s rows from %s config=%s",
            config_rows,
            source.name,
            config_name or "default",
        )
        if len(rows) >= sample_cap:
            break

    write_jsonl(rows, output_path)
    logger.info("Saved %s rows from %s to %s", len(rows), source.name, output_path)
    return len(rows), None


def _build_oasst_paths(
    rows: list[dict[str, Any]],
    source: DatasetSource,
    sample_cap: int,
) -> list[dict[str, Any]]:
    id_to_node = {row["message_id"]: row for row in rows if row.get("message_id")}
    normalized_rows: list[dict[str, Any]] = []
    seen_hashes: set[tuple[str, ...]] = set()

    for row in rows:
        role = _normalize_role(str(row.get("role", "")), source)
        if role != "assistant":
            continue

        path_nodes: list[dict[str, Any]] = []
        current = row
        while current is not None:
            path_nodes.append(current)
            parent_id = current.get("parent_id")
            current = id_to_node.get(parent_id)
        path_nodes.reverse()

        messages: list[dict[str, str]] = []
        for node in path_nodes:
            mapped_role = _normalize_role(str(node.get("role", "")), source)
            content = _normalize_scalar(node.get("text"))
            if mapped_role not in {"user", "assistant"} or not content:
                continue
            if messages and messages[-1]["role"] == mapped_role:
                messages[-1]["content"] = f"{messages[-1]['content']}\n\n{content}"
            else:
                messages.append({"role": mapped_role, "content": content})

        if source.max_messages_per_example and len(messages) > source.max_messages_per_example:
            messages = messages[-source.max_messages_per_example :]
            while messages and messages[0]["role"] == "assistant":
                messages = messages[1:]

        if len(messages) < 2 or messages[-1]["role"] != "assistant":
            continue

        path_hash = tuple(f"{message['role']}::{message['content']}" for message in messages)
        if path_hash in seen_hashes:
            continue
        seen_hashes.add(path_hash)

        normalized_rows.append(
            {
                "id": str(row.get("message_id", f"{source.name}-{len(normalized_rows)}")),
                "record_type": "conversation",
                "source": source.name,
                "license": source.license,
                "citation": source.citation,
                "url": source.url,
                "messages": messages,
                "topic": source.topic_hint,
                "difficulty": None,
                "metadata": {
                    "dataset_id": source.dataset_id,
                    "message_tree_id": row.get("message_tree_id"),
                },
            }
        )
        if len(normalized_rows) >= sample_cap:
            break

    return normalized_rows


def _download_oasst_tree_dataset(
    source: DatasetSource,
    output_path: Path,
    max_samples: int,
    logger,
) -> tuple[int, str | None]:
    from datasets import load_dataset

    dataset = load_dataset(
        path=source.dataset_id,
        name=source.config_name,
        split=source.split or "train",
        streaming=False,
    )

    candidate_rows: list[dict[str, Any]] = []
    for example in dataset:
        row = dict(example)
        if not _record_matches_filters(source, row):
            continue
        if row.get("deleted") is True:
            continue
        if not _normalize_scalar(row.get("text")):
            continue
        candidate_rows.append(row)

    sample_cap = int(source.max_samples or max_samples)
    rows = _build_oasst_paths(candidate_rows, source=source, sample_cap=sample_cap)
    write_jsonl(rows, output_path)
    logger.info("Saved %s reconstructed conversation paths from %s", len(rows), source.name)
    return len(rows), None


def _extract_pdf_records(
    pdf_path: Path,
    source: DatasetSource,
) -> list[dict[str, Any]]:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    rows: list[dict[str, Any]] = []
    for page_index, page in enumerate(reader.pages):
        page_text = (page.extract_text() or "").strip()
        if not page_text:
            continue
        rows.append(
            {
                "id": f"{source.name}-page-{page_index + 1}",
                "record_type": "text",
                "source": source.name,
                "license": source.license,
                "citation": source.citation,
                "url": source.url,
                "title": f"{source.name.replace('_', ' ').title()} page {page_index + 1}",
                "text": page_text,
                "topic": source.topic_hint,
                "metadata": {"page": page_index + 1, "asset_path": str(pdf_path)},
            }
        )
    return rows


def _find_openstax_pdf_url(book_page_url: str) -> str | None:
    response = requests.get(book_page_url, timeout=60)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        href_lower = href.lower()
        label = anchor.get_text(" ", strip=True).lower()
        if href_lower.endswith(".pdf") or ("pdf" in href_lower and "download" in label):
            return urljoin(book_page_url, href)
    match = re.search(r"https?://[^\s\"']+\.pdf", response.text, flags=re.IGNORECASE)
    if match:
        return match.group(0)
    return None


def _download_openstax_dataset(
    source: DatasetSource,
    assets_dir: Path,
    output_path: Path,
    logger,
) -> tuple[int, str | None]:
    pdf_url = _find_openstax_pdf_url(source.url)
    if not pdf_url:
        return 0, "Could not find a PDF link on the OpenStax page."

    pdf_path = assets_dir / f"{source.name}.pdf"
    response = requests.get(pdf_url, timeout=120)
    response.raise_for_status()
    pdf_path.write_bytes(response.content)

    rows = _extract_pdf_records(pdf_path, source)
    write_jsonl(rows, output_path)
    logger.info("Downloaded %s and extracted %s page records", pdf_url, len(rows))
    return len(rows), None


def _strip_gutenberg_boilerplate(text: str) -> str:
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "***START OF THE PROJECT GUTENBERG EBOOK",
    ]
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "***END OF THE PROJECT GUTENBERG EBOOK",
    ]
    for marker in start_markers:
        if marker in text:
            text = text.split(marker, 1)[1]
            break
    for marker in end_markers:
        if marker in text:
            text = text.split(marker, 1)[0]
            break
    return text.strip()


def _resolve_gutenberg_text_url(page_url: str) -> str:
    if page_url.lower().endswith(".txt"):
        return page_url
    response = requests.get(page_url, timeout=60)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        if ".txt" in href and "utf-8" in href.lower():
            return urljoin(page_url, href)
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        if href.lower().endswith(".txt"):
            return urljoin(page_url, href)

    ebook_id_match = re.search(r"/ebooks/(\d+)", page_url)
    if not ebook_id_match:
        raise ValueError(f"Could not infer Project Gutenberg ebook id from {page_url}")
    ebook_id = ebook_id_match.group(1)
    return f"https://www.gutenberg.org/cache/epub/{ebook_id}/pg{ebook_id}.txt"


def _download_gutenberg_dataset(
    source: DatasetSource,
    output_path: Path,
    logger,
) -> tuple[int, str | None]:
    text_url = _resolve_gutenberg_text_url(source.url)
    response = requests.get(text_url, timeout=120)
    response.raise_for_status()
    book_text = _strip_gutenberg_boilerplate(response.text)

    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", book_text) if paragraph.strip()]
    rows = []
    for index, paragraph in enumerate(paragraphs):
        if len(paragraph) < 250:
            continue
        rows.append(
            {
                "id": f"{source.name}-{index + 1}",
                "record_type": "text",
                "source": source.name,
                "license": source.license,
                "citation": source.citation,
                "url": text_url,
                "title": source.name.replace("_", " ").title(),
                "text": paragraph,
                "topic": source.topic_hint,
                "metadata": {"paragraph_index": index + 1},
            }
        )
    write_jsonl(rows, output_path)
    logger.info("Downloaded Project Gutenberg text from %s with %s usable paragraphs", text_url, len(rows))
    return len(rows), None


def _download_arxiv_dataset(
    source: DatasetSource,
    output_path: Path,
    max_samples: int,
    logger,
) -> tuple[int, str | None]:
    params = {
        "search_query": source.query or "cat:physics*",
        "start": 0,
        "max_results": int(source.max_samples or max_samples),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    response = requests.get(source.url, params=params, timeout=120)
    response.raise_for_status()

    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    root = ET.fromstring(response.text)

    rows: list[dict[str, Any]] = []
    for entry in root.findall("atom:entry", ns):
        license_url = entry.findtext("arxiv:license", default="", namespaces=ns).strip()
        if source.require_explicit_license:
            allowed = set(source.allowed_licenses or [])
            if not license_url or (allowed and license_url not in allowed):
                continue

        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
        summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
        paper_url = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
        primary_category = entry.find("arxiv:primary_category", ns)
        category_term = primary_category.attrib.get("term", "") if primary_category is not None else ""

        rows.append(
            {
                "id": paper_url or f"{source.name}-{len(rows) + 1}",
                "record_type": "text",
                "source": source.name,
                "license": license_url or source.license,
                "citation": source.citation,
                "url": paper_url,
                "title": title,
                "text": summary,
                "topic": source.topic_hint,
                "metadata": {"category": category_term},
            }
        )
        if len(rows) >= int(source.max_samples or max_samples):
            break

    write_jsonl(rows, output_path)
    logger.info("Saved %s arXiv abstract records to %s", len(rows), output_path)
    return len(rows), None


def _collect_custom_manifest(project_root: Path, source: DatasetSource) -> dict[str, Any]:
    matched = sorted(glob.glob(str(project_root / (source.path_glob or "data/custom/*.jsonl"))))
    return {
        "source": source.to_manifest_dict(),
        "status": "local-files",
        "matched_files": matched,
        "record_count": None,
    }


def download_enabled_sources(
    config: dict[str, Any],
    logger,
    only_sources: list[str] | None = None,
) -> dict[str, Any]:
    project_root = Path(config["__project_root__"])
    raw_dir = ensure_dir(project_root / config.get("raw_data_dir", "data/raw"))
    assets_dir = ensure_dir(raw_dir / "assets")
    manifest_path = project_root / config.get("dataset_manifest_path", "data/raw/dataset_manifest.json")
    max_samples = int(config.get("max_samples_per_dataset", 50000))
    selected = set(only_sources or [])

    manifest: dict[str, Any] = {
        "project_name": config.get("project_name", "physics-chatbot"),
        "max_samples_per_dataset": max_samples,
        "sources": [],
    }

    for source in load_sources_from_config(config):
        if selected and source.name not in selected:
            continue
        if not source.enabled:
            manifest["sources"].append(
                {
                    "source": source.to_manifest_dict(),
                    "status": "disabled",
                    "record_count": 0,
                    "output_path": None,
                }
            )
            continue
        if not _is_known_license(source.license):
            manifest["sources"].append(
                {
                    "source": source.to_manifest_dict(),
                    "status": "skipped",
                    "record_count": 0,
                    "output_path": None,
                    "reason": "Unknown or missing license.",
                }
            )
            logger.warning("Skipping %s because its license is unknown or missing.", source.name)
            continue
        if source.type == "custom_glob":
            manifest["sources"].append(_collect_custom_manifest(project_root, source))
            continue

        output_path = raw_dir / f"{source.name}.jsonl"
        count = 0
        error: str | None = None

        try:
            if source.type == "hf":
                count, error = _download_hf_dataset(source, output_path=output_path, max_samples=max_samples, logger=logger)
            elif source.type == "hf_oasst_tree":
                count, error = _download_oasst_tree_dataset(
                    source,
                    output_path=output_path,
                    max_samples=max_samples,
                    logger=logger,
                )
            elif source.type == "openstax":
                count, error = _download_openstax_dataset(source, assets_dir=assets_dir, output_path=output_path, logger=logger)
            elif source.type == "gutenberg":
                count, error = _download_gutenberg_dataset(source, output_path=output_path, logger=logger)
            elif source.type == "arxiv":
                count, error = _download_arxiv_dataset(source, output_path=output_path, max_samples=max_samples, logger=logger)
            else:
                error = f"Unsupported source type: {source.type}"
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            error = str(exc)
            logger.exception("Failed to download %s", source.name)

        manifest["sources"].append(
            {
                "source": source.to_manifest_dict(),
                "status": "ok" if not error else "error",
                "record_count": count,
                "output_path": str(output_path) if output_path.exists() else None,
                "reason": error,
            }
        )

    save_json(manifest, manifest_path)
    logger.info("Wrote dataset manifest to %s", manifest_path)
    return manifest
