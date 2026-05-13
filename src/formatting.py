from __future__ import annotations

import re
from typing import Any


PHYSICS_TOPICS = [
    "mechanics",
    "electromagnetism",
    "thermodynamics",
    "optics",
    "waves",
    "relativity",
    "quantum physics",
    "nuclear physics",
    "astrophysics",
    "mathematical physics",
    "experimental physics",
    "general physics",
]


TOPIC_KEYWORDS = {
    "mechanics": ["force", "motion", "velocity", "acceleration", "momentum", "newton", "torque", "friction"],
    "electromagnetism": ["electric", "magnetic", "voltage", "current", "charge", "field", "circuit", "faraday"],
    "thermodynamics": ["heat", "temperature", "entropy", "thermo", "pressure", "engine", "ideal gas"],
    "optics": ["light", "lens", "mirror", "refraction", "reflection", "interference", "diffraction", "optics"],
    "waves": ["wave", "frequency", "wavelength", "sound", "oscillation", "amplitude", "resonance"],
    "relativity": ["relativity", "einstein", "spacetime", "time dilation", "lorentz", "mass-energy"],
    "quantum physics": ["quantum", "wavefunction", "uncertainty", "photon", "schrodinger", "spin", "orbital"],
    "nuclear physics": ["nuclear", "radioactive", "decay", "fission", "fusion", "isotope", "half-life"],
    "astrophysics": ["star", "galaxy", "cosmology", "planet", "black hole", "supernova", "astrophysics"],
    "mathematical physics": ["lagrangian", "hamiltonian", "tensor", "differential equation", "calculus", "symmetry"],
    "experimental physics": ["measurement", "uncertainty", "instrument", "detector", "lab", "experimental"],
}


DEFAULT_SYSTEM_PROMPT = (
    "You are PhysicsGPT, a precise physics tutor. Explain concepts step by step, "
    "show formulas, define variables, use SI units, and warn when information is uncertain."
)


def topic_from_text(text: str, topic_hint: str | None = None) -> str:
    if topic_hint and topic_hint in PHYSICS_TOPICS:
        return topic_hint

    lowered = (text or "").lower()
    best_topic = "general physics"
    best_score = 0

    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in lowered)
        if score > best_score:
            best_topic = topic
            best_score = score

    return best_topic


def difficulty_from_text(question: str, answer: str) -> str:
    combined = f"{question} {answer}".lower()
    if len(answer) > 1000 or any(keyword in combined for keyword in ["derive", "tensor", "quantum field", "relativistic"]):
        return "advanced"
    if len(answer) > 300 or any(keyword in combined for keyword in ["calculate", "solve", "equation", "proof"]):
        return "intermediate"
    return "beginner"


def build_messages(user_text: str, assistant_text: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text.strip()},
        {"role": "assistant", "content": assistant_text.strip()},
    ]


def make_text_prompt(title: str | None, topic: str) -> str:
    if title:
        return (
            f"Teach me this physics topic clearly: {title}. "
            "Explain the core idea step by step, define symbols, and keep SI units consistent."
        )
    return (
        f"Teach me an important concept in {topic}. "
        "Explain it step by step, define symbols, and keep SI units consistent."
    )


def format_qa_example(record: dict[str, Any], system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> dict[str, Any]:
    question = str(record.get("question", "")).strip()
    answer = str(record.get("answer", "")).strip()
    topic = topic_from_text(f"{question}\n{answer}", topic_hint=record.get("topic"))
    difficulty = str(record.get("difficulty") or difficulty_from_text(question, answer))
    return {
        "messages": build_messages(question, answer, system_prompt=system_prompt),
        "source": record.get("source", "unknown"),
        "topic": topic,
        "difficulty": difficulty,
        "license": record.get("license", "unknown"),
    }


def format_text_example(
    record: dict[str, Any],
    text_chunk: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> dict[str, Any]:
    title = str(record.get("title") or "").strip() or None
    topic = topic_from_text(f"{title or ''}\n{text_chunk}", topic_hint=record.get("topic"))
    assistant_text = text_chunk.strip()
    prompt = make_text_prompt(title, topic)
    difficulty = str(record.get("difficulty") or difficulty_from_text(prompt, assistant_text))
    return {
        "messages": build_messages(prompt, assistant_text, system_prompt=system_prompt),
        "source": record.get("source", "unknown"),
        "topic": topic,
        "difficulty": difficulty,
        "license": record.get("license", "unknown"),
    }


def _normalize_multimodal_messages(messages: list[dict[str, Any]], model_name: str) -> list[dict[str, Any]]:
    lowered = model_name.lower()
    needs_multimodal_text_blocks = "gemma-3" in lowered or "qwen3.5" in lowered
    if not needs_multimodal_text_blocks:
        return messages

    normalized: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            normalized.append(
                {
                    "role": message.get("role", "user"),
                    "content": [{"type": "text", "text": content}],
                }
            )
        else:
            normalized.append(message)
    return normalized


def messages_to_text(
    messages: list[dict[str, Any]],
    tokenizer: Any,
    model_name: str,
    add_generation_prompt: bool = False,
    enable_thinking: bool = False,
) -> str:
    normalized_messages = _normalize_multimodal_messages(messages, model_name=model_name)

    if hasattr(tokenizer, "apply_chat_template"):
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
        }
        try:
            return tokenizer.apply_chat_template(
                normalized_messages,
                enable_thinking=enable_thinking,
                **kwargs,
            )
        except TypeError:
            return tokenizer.apply_chat_template(normalized_messages, **kwargs)

    rendered_parts: list[str] = []
    for message in messages:
        role = message.get("role", "user").upper()
        content = message.get("content", "")
        if isinstance(content, list):
            text_fragments = [part.get("text", "") for part in content if isinstance(part, dict)]
            content = " ".join(fragment for fragment in text_fragments if fragment)
        rendered_parts.append(f"{role}: {content}")
    if add_generation_prompt:
        rendered_parts.append("ASSISTANT:")
    return "\n".join(rendered_parts)


def strip_thinking_tokens(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    return re.sub(r"\n{3,}", "\n\n", text).strip()
