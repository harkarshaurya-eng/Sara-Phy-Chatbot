#!/usr/bin/env bash
set -euo pipefail

python src/train_tokenizer.py --config configs/tiny_gpt.yaml "$@"
