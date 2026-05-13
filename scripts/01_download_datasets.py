from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_sources import download_enabled_sources, load_sources_from_config
from src.train_utils import load_config, resolve_path, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download enabled physics datasets with license-aware manifesting.")
    parser.add_argument("--config", default="config.yaml", help="Path to the YAML config file.")
    parser.add_argument("--only-source", action="append", help="Limit downloading to one or more source names.")
    parser.add_argument("--list-sources", action="store_true", help="List configured dataset sources and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    log_path = resolve_path(PROJECT_ROOT, config.get("log_dir", "outputs/logs")) / "download_datasets.log"
    logger = setup_logging("download_datasets", log_file=log_path)

    if args.list_sources:
        sources = [source.to_manifest_dict() for source in load_sources_from_config(config)]
        print(json.dumps(sources, indent=2))
        return

    manifest = download_enabled_sources(config=config, logger=logger, only_sources=args.only_source)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
