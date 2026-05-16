#!/usr/bin/env bash
set -euo pipefail

python src/tokenize_dataset.py --config configs/tiny_gpt.yaml
python src/train.py --model_config configs/tiny_gpt.yaml --train_config configs/train_config.yaml "$@"
