from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api_server import create_app
from src.train_utils import get_system_prompt, load_config, resolve_path, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the physics chatbot via a local OpenAI-compatible API.")
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML config file.")
    parser.add_argument("--adapter", default="outputs/adapters/final", help="Path to the LoRA adapter.")
    parser.add_argument("--merged-model", default=None, help="Path to a merged model directory.")
    parser.add_argument("--base-model", default=None, help="Optional override for the base model.")
    parser.add_argument("--host", default=None, help="Override API host.")
    parser.add_argument("--port", type=int, default=None, help="Override API port.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    logger = setup_logging(
        "serve_openai_api",
        log_file=resolve_path(PROJECT_ROOT, config.get("log_dir", "outputs/logs")) / "serve_openai_api.log",
    )

    api_cfg = config.get("api", {})
    base_model = str(args.base_model or config.get("base_model"))
    merged_model_path = resolve_path(PROJECT_ROOT, args.merged_model) if args.merged_model else None
    adapter_path = None if merged_model_path else resolve_path(PROJECT_ROOT, args.adapter)
    if adapter_path and not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

    app = create_app(
        base_model=base_model,
        served_model_name=str(api_cfg.get("model_name", "physics-chatbot")),
        system_prompt=get_system_prompt(config),
        adapter_path=str(adapter_path) if adapter_path else None,
        merged_model_path=str(merged_model_path) if merged_model_path else None,
        trust_remote_code=bool(config.get("trust_remote_code", True)),
        logger=logger,
    )

    host = args.host or api_cfg.get("host", "127.0.0.1")
    port = args.port or int(api_cfg.get("port", 8000))
    logger.info("Starting local API on http://%s:%s", host, port)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
