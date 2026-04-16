#!/bin/bash
# Download WikiText-103 (HF) into a single text file for prepare_text.sh.
set -euo pipefail
OUT="${1:-$HOME/datasets/wiki/wikitext-103/wiki.train.tokens}"
mkdir -p "$(dirname "$OUT")"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$REPO_ROOT/unsupervised_wav/venv/bin/activate"
python -c "import datasets" 2>/dev/null || pip install -q datasets
python "$REPO_ROOT/scripts/dump_wikitext103_train.py" --out "$OUT"
wc -l -c "$OUT"
