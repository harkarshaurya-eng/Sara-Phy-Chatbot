from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import yaml

try:
    import numpy as np
except Exception:  # pragma: no cover - optional at parse time
    np = None

try:
    import torch
except Exception:  # pragma: no cover - optional at parse time
    torch = None


def project_root_from_file(file_path: str | Path) -> Path:
    return Path(file_path).resolve().parents[1]


def resolve_path(project_root: str | Path, value: str | Path) -> Path:
    value_path = Path(value)
    if value_path.is_absolute():
        return value_path
    return Path(project_root).resolve() / value_path


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def ensure_parent_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    return path_obj.parent


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    with config_file.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config["__config_path__"] = str(config_file)
    config["__project_root__"] = str(config_file.parent)
    return config


def save_json(payload: Any, output_path: str | Path) -> None:
    ensure_parent_dir(output_path)
    with Path(output_path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def read_jsonl(input_path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path_obj = Path(input_path)
    if not path_obj.exists():
        return rows
    with path_obj.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path_obj} at line {line_number}: {exc}") from exc
    return rows


def write_jsonl(records: list[dict[str, Any]], output_path: str | Path) -> None:
    ensure_parent_dir(output_path)
    with Path(output_path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def setup_logging(name: str, log_file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        ensure_parent_dir(log_file)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def detect_runtime() -> dict[str, Any]:
    runtime: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch_available": torch is not None,
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_device_name": None,
        "mps_available": False,
        "tpu_available": False,
        "xla_supported_devices": [],
        "xla_hardware": None,
        "accelerator": "cpu",
        "bf16_supported": False,
    }

    if torch is not None:
        runtime["cuda_available"] = bool(torch.cuda.is_available())
        runtime["cuda_device_count"] = int(torch.cuda.device_count())
        runtime["cuda_device_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        runtime["mps_available"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        if torch.cuda.is_available():
            runtime["accelerator"] = "cuda"
            runtime["bf16_supported"] = bool(
                hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
            )
        elif runtime["mps_available"]:
            runtime["accelerator"] = "mps"

    # Modern TPU detection using PJRT (compatible with TPU v5e)
    try:
        import torch_xla.core.xla_model as xm

        # Try the modern API first (torch_xla 2.5+)
        try:
            device = xm.xla_device()
            # Validate by creating a test tensor
            test_tensor = torch.zeros(1, device=device)
            del test_tensor
            runtime["tpu_available"] = True
            runtime["accelerator"] = "tpu"
            runtime["xla_hardware"] = xm.xla_device_hw(device)
            runtime["bf16_supported"] = True
            # Try to get supported devices list (may not exist in all versions)
            try:
                supported = xm.get_xla_supported_devices(devkind=None)
                runtime["xla_supported_devices"] = supported
            except Exception:
                runtime["xla_supported_devices"] = [str(device)]
        except Exception:
            # Fallback to legacy API
            try:
                supported = xm.get_xla_supported_devices(devkind=None)
                if supported:
                    runtime["tpu_available"] = True
                    runtime["xla_supported_devices"] = supported
                    runtime["accelerator"] = "tpu"
                    device = xm.xla_device()
                    runtime["xla_hardware"] = xm.xla_device_hw(device)
                    runtime["bf16_supported"] = True
            except Exception:
                pass
    except Exception:
        pass

    # Collect TPU memory info if available
    if runtime["tpu_available"]:
        try:
            mem_info = get_tpu_memory_info()
            if mem_info:
                runtime["tpu_memory"] = mem_info
        except Exception:
            pass

    return runtime


def get_tpu_memory_info() -> dict[str, Any] | None:
    """Get TPU memory usage information. Returns None if not on TPU."""
    try:
        import torch_xla.core.xla_model as xm

        device = xm.xla_device()
        info = xm.get_memory_info(device)
        total_bytes = info.get("kb_total", 0) * 1024
        free_bytes = info.get("kb_free", 0) * 1024
        used_bytes = total_bytes - free_bytes
        return {
            "total_gb": round(total_bytes / (1024**3), 2),
            "used_gb": round(used_bytes / (1024**3), 2),
            "free_gb": round(free_bytes / (1024**3), 2),
            "utilization_pct": round((used_bytes / max(total_bytes, 1)) * 100, 1),
        }
    except Exception:
        return None


def format_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    if minutes < 60:
        return f"{minutes}m {remaining_seconds:.0f}s"
    hours = int(minutes // 60)
    remaining_minutes = minutes % 60
    return f"{hours}h {remaining_minutes}m {remaining_seconds:.0f}s"


def collect_package_versions(packages: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def model_architecture_hint(model_name: str) -> str:
    lowered = model_name.lower()
    if "gemma-3" in lowered or "qwen3.5" in lowered:
        return "image_text_to_text"
    return "causal_lm"


def supports_4bit_quantization(model_name: str, runtime: dict[str, Any]) -> bool:
    return runtime.get("accelerator") == "cuda" and model_architecture_hint(model_name) == "causal_lm"


def default_dtype(runtime: dict[str, Any], prefer_bf16: bool = True):
    if torch is None:
        return None
    if runtime.get("accelerator") in {"tpu", "cuda"}:
        if prefer_bf16 and runtime.get("bf16_supported", False):
            return torch.bfloat16
        if runtime.get("accelerator") == "cuda":
            return torch.float16
    return torch.float32


def get_system_prompt(config: dict[str, Any]) -> str:
    return config.get("inference", {}).get(
        "system_prompt",
        (
            "You are PhysicsGPT, a precise physics tutor. Explain concepts step by step, "
            "show formulas, define variables, use SI units, and warn when information is uncertain."
        ),
    )


def now_ts() -> int:
    return int(time.time())


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    rendered_rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        rendered_rows.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(rendered_rows)


def write_text(text: str, output_path: str | Path) -> None:
    ensure_parent_dir(output_path)
    Path(output_path).write_text(text, encoding="utf-8")


def append_jsonl(record: dict[str, Any], output_path: str | Path) -> None:
    ensure_parent_dir(output_path)
    with Path(output_path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
