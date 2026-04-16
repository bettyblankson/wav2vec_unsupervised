#!/bin/bash
# Re-run only text preparation + GAN training (e.g. after fixing empty wiki.train.tokens).
# Skips manifests, rVAD, silence removal, and prepare_audio — those stay checkpointed.
#
# Export the same variables you use for run_training_subset.sh:
#   RUN_TAG, MAX_UNLABELLED_TEXT_LINES, GAN_MAX_UPDATE
#
# Usage:
#   export RUN_TAG=run_500_audio MAX_UNLABELLED_TEXT_LINES=3000 GAN_MAX_UPDATE=10000  # 500 utt×20; use 20k/40k for 1k/2k train
#   bash refresh_text_and_train_gans.sh \
#     "$HOME/datasets/librispeech/split_pool/train" \
#     "$HOME/datasets/librispeech/split_pool/val" \
#     "$HOME/datasets/librispeech/split_pool/test" \
#     "$HOME/datasets/wiki/wikitext-103/wiki.train.tokens"

set -euo pipefail

if [ -z "${BASH_VERSION:-}" ]; then
  echo "Use bash to run this script." >&2
  exit 1
fi

if [ "$#" -lt 4 ]; then
  echo "Usage: $0 TRAIN_WAV_DIR VAL_WAV_DIR TEST_WAV_DIR UNLABELLED_TEXT_FILE" >&2
  exit 1
fi

UNLABELLED_TEXT_FILE="$4"
if [[ ! -f "$UNLABELLED_TEXT_FILE" ]]; then
  echo "ERROR: text file not found: $UNLABELLED_TEXT_FILE" >&2
  exit 1
fi
if [[ ! -s "$UNLABELLED_TEXT_FILE" ]]; then
  echo "ERROR: text file is empty (downloaded failed?): $UNLABELLED_TEXT_FILE" >&2
  echo "Fix: re-download WikiText (non-zero size), then re-run this script." >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

source "$REPO_ROOT/wav2vec_functions.sh" "$@"

CHECKPOINT_FILE="$REPO_ROOT/unsupervised_wav/data/checkpoints/librispeech/progress.checkpoint"
mkdir -p "$(dirname "$CHECKPOINT_FILE")"
touch "$CHECKPOINT_FILE"

for step in prepare_text train_gans; do
  sed -i "/^${step}:COMPLETED\$/d" "$CHECKPOINT_FILE" 2>/dev/null || true
  sed -i "/^${step}:IN_PROGRESS\$/d" "$CHECKPOINT_FILE" 2>/dev/null || true
done

create_dirs
activate_venv
setup_path

log "refresh_text_and_train_gans: removing $TEXT_OUTPUT (regenerate phones + LM binaries)"
rm -rf "$TEXT_OUTPUT"

create_dirs
prepare_text

source "$REPO_ROOT/gans_functions.sh"
train_gans

log "refresh_text_and_train_gans: done (RUN_TAG=${RUN_TAG:-?})"
