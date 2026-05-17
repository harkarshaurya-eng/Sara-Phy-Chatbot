#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f "requirements.txt" ]]; then
  echo "requirements.txt not found. First cd into the repo root, for example: /content/physics-gpt-from-scratch"
  exit 1
fi

python -m pip install --upgrade pip
pip install -r requirements.txt
