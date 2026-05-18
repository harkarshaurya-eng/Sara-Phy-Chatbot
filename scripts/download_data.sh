#!/usr/bin/env bash
set -euo pipefail

python src/download_datasets.py \
  --only-group physics \
  --only-group conversation \
  --max-samples-per-dataset 10000 \
  "$@"
