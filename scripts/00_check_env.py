from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train_utils import collect_package_versions, detect_runtime, load_config, resolve_path, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check local or Colab training environment for PhysicsGPT.")
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML config file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    log_path = resolve_path(PROJECT_ROOT, config.get("log_dir", "outputs/logs")) / "check_env.log"
    logger = setup_logging("check_env", log_file=log_path)

    runtime = detect_runtime()
    versions = collect_package_versions(
        [
            "torch",
            "torch_xla",
            "transformers",
            "datasets",
            "accelerate",
            "peft",
            "trl",
            "bitsandbytes",
            "fastapi",
            "uvicorn",
            "tensorboard",
            "evaluate",
        ]
    )

    summary = {
        "project_root": str(PROJECT_ROOT),
        "config_path": config["__config_path__"],
        "hf_token_present": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")),
        "runtime": runtime,
        "package_versions": versions,
    }

    logger.info("Environment summary:\n%s", json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
