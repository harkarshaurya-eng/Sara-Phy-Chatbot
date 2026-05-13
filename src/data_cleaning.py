from __future__ import annotations

import hashlib
import html
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any


TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
URL_RE = re.compile(r"https?://\S+")
PUNCT_ONLY_RE = re.compile(r"^[\W_]+$")


def normalize_text(text: Any, remove_html: bool = True) -> str:
    if text is None:
        return ""
    text = str(text)
    text = html.unescape(text)
    if remove_html:
        text = TAG_RE.sub(" ", text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    text = SPACE_RE.sub(" ", text).strip()
    return text


def shingle_tokens(text: str, size: int = 3) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    if len(tokens) <= size:
        return [" ".join(tokens)] if tokens else []
    return [" ".join(tokens[index : index + size]) for index in range(len(tokens) - size + 1)]


def simhash(text: str, bits: int = 64) -> int:
    shingles = shingle_tokens(text)
    if not shingles:
        return 0

    vector = [0] * bits
    for shingle in shingles:
        digest = hashlib.md5(shingle.encode("utf-8")).hexdigest()
        value = int(digest, 16)
        for bit_idx in range(bits):
            if value & (1 << bit_idx):
                vector[bit_idx] += 1
            else:
                vector[bit_idx] -= 1

    signature = 0
    for bit_idx, weight in enumerate(vector):
        if weight >= 0:
            signature |= 1 << bit_idx
    return signature


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


class SimHashDeduper:
    def __init__(self, threshold: int = 3, bands: int = 4) -> None:
        self.threshold = threshold
        self.bands = bands
        self.band_buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        self.signatures: list[int] = []

    def _band_keys(self, signature: int) -> list[tuple[int, int]]:
        bits_per_band = math.ceil(64 / self.bands)
        keys: list[tuple[int, int]] = []
        mask = (1 << bits_per_band) - 1
        for band_idx in range(self.bands):
            keys.append((band_idx, (signature >> (band_idx * bits_per_band)) & mask))
        return keys

    def is_duplicate(self, text: str) -> bool:
        signature = simhash(text)
        candidate_indexes: set[int] = set()
        for key in self._band_keys(signature):
            candidate_indexes.update(self.band_buckets.get(key, []))
        for index in candidate_indexes:
            if hamming_distance(signature, self.signatures[index]) <= self.threshold:
                return True
        self.signatures.append(signature)
        for key in self._band_keys(signature):
            self.band_buckets[key].append(len(self.signatures) - 1)
        return False


def is_low_quality(question: str, answer: str, min_question_chars: int, min_answer_chars: int) -> bool:
    question = normalize_text(question)
    answer = normalize_text(answer)

    if not question or not answer:
        return True
    if len(question) < min_question_chars or len(answer) < min_answer_chars:
        return True
    if PUNCT_ONLY_RE.match(question) or PUNCT_ONLY_RE.match(answer):
        return True
    if answer.lower() in {"n/a", "unknown", "no idea", "not sure"}:
        return True
    if len(set(answer.lower().split())) < 5:
        return True
    if URL_RE.fullmatch(answer):
        return True
    return False


def split_into_text_chunks(text: str, min_chars: int = 300, max_chars: int = 1800) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        projected_len = current_len + len(sentence) + 1
        if current and projected_len > max_chars:
            chunk = " ".join(current).strip()
            if len(chunk) >= min_chars:
                chunks.append(chunk)
            current = [sentence]
            current_len = len(sentence)
        else:
            current.append(sentence)
            current_len = projected_len

    trailing = " ".join(current).strip()
    if len(trailing) >= min_chars:
        chunks.append(trailing)

    return chunks


def exact_dedupe_key(messages: list[dict[str, Any]], source: str | None = None) -> str:
    flattened = "\n".join(
        f"{message.get('role', '')}:{normalize_text(message.get('content', ''))}" for message in messages
    )
    return hashlib.sha1(flattened.encode("utf-8")).hexdigest()


def shuffle_and_split(
    records: list[dict[str, Any]],
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not math.isclose(train_ratio + validation_ratio + test_ratio, 1.0, abs_tol=1e-6):
        raise ValueError("Train/validation/test ratios must add up to 1.0.")

    randomized = list(records)
    random.Random(seed).shuffle(randomized)

    train_end = int(len(randomized) * train_ratio)
    validation_end = train_end + int(len(randomized) * validation_ratio)
    train = randomized[:train_end]
    validation = randomized[train_end:validation_end]
    test = randomized[validation_end:]
    return train, validation, test


def topic_counts(records: list[dict[str, Any]]) -> Counter:
    return Counter(record.get("topic", "unknown") for record in records)


def source_counts(records: list[dict[str, Any]]) -> Counter:
    return Counter(record.get("source", "unknown") for record in records)
