#!/bin/bash
# Generic subset runner for wav2vec-u experiments.
# Runs wav2vec prep + GAN training with explicit caps and a dedicated RUN_TAG.
#
# Usage:
#   bash run_training_subset.sh \
#     TRAIN_WAV_DIR VAL_WAV_DIR TEST_WAV_DIR UNLABELLED_TEXT_FILE \
#     TRAIN_CAP VAL_CAP TEXT_CAP RUN_TAG [GAN_MAX_UPDATE]
#
# If GAN_MAX_UPDATE is omitted, it defaults to (TRAIN_CAP * 20): 500→10000, 1000→20000, 2000→40000.
#
# Example:
#   bash run_training_subset.sh \
#     "$HOME/datasets/librispeech/train-clean-100-wav" \
#     "$HOME/datasets/librispeech/dev-clean-wav" \
#     "$HOME/datasets/librispeech/test-clean-wav" \
#     "$HOME/datasets/wiki/wiki.train.tokens" \
#     500 50 3000 run_500_audio

set -euo pipefail

if [ -z "${BASH_VERSION:-}" ]; then
  echo "Use bash to run this script." >&2
  exit 1
fi

if [ "$#" -lt 8 ]; then
  echo "Usage: $0 TRAIN_WAV_DIR VAL_WAV_DIR TEST_WAV_DIR UNLABELLED_TEXT_FILE TRAIN_CAP VAL_CAP TEXT_CAP RUN_TAG [GAN_MAX_UPDATE]" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

TRAIN_WAV_DIR="$1"
VAL_WAV_DIR="$2"
TEST_WAV_DIR="$3"
UNLABELLED_TEXT_FILE="$4"
TRAIN_CAP="$5"
VAL_CAP="$6"
TEXT_CAP="$7"
RUN_TAG_ARG="$8"
# Scale max_update with data size unless overridden: 500→10k, 1k→20k, 2k→40k (20 updates per train utterance).
if [ "$#" -ge 9 ] && [ -n "${9:-}" ]; then
  GAN_MAX_UPDATE_ARG="$9"
else
  GAN_MAX_UPDATE_ARG=$((TRAIN_CAP * 20))
fi

export MAX_TRAIN_UTTERANCES="$TRAIN_CAP"
export MAX_VALID_UTTERANCES="$VAL_CAP"
export MAX_UNLABELLED_TEXT_LINES="$TEXT_CAP"
export GAN_MAX_UPDATE="$GAN_MAX_UPDATE_ARG"
export RUN_TAG="$RUN_TAG_ARG"

CHECKPOINT_FILE="$REPO_ROOT/unsupervised_wav/data/checkpoints/librispeech/progress.checkpoint"
mkdir -p "$(dirname "$CHECKPOINT_FILE")"
touch "$CHECKPOINT_FILE"

# Clear checkpointed steps that depend on caps/data or should produce a fresh run.
for step in create_manifests_train create_manifests_val create_manifests_test \
            subsample_manifests create_rVADfast remove_silence \
            create_manifests_nonsil_train create_manifests_nonsil_val \
            prepare_audio prepare_text train_gans; do
  sed -i "/^${step}:COMPLETED\$/d" "$CHECKPOINT_FILE" 2>/dev/null || true
  sed -i "/^${step}:IN_PROGRESS\$/d" "$CHECKPOINT_FILE" 2>/dev/null || true
done

echo "[run_training_subset] RUN_TAG=$RUN_TAG"
echo "[run_training_subset] caps: train=$MAX_TRAIN_UTTERANCES val=$MAX_VALID_UTTERANCES text_lines=$MAX_UNLABELLED_TEXT_LINES max_update=$GAN_MAX_UPDATE"

bash "$REPO_ROOT/run_wav2vec.sh" \
  "$TRAIN_WAV_DIR" \
  "$VAL_WAV_DIR" \
  "$TEST_WAV_DIR" \
  "$UNLABELLED_TEXT_FILE"

bash "$REPO_ROOT/run_gans.sh"

RUN_DIR="$REPO_ROOT/unsupervised_wav/data/results/librispeech/runs/$RUN_TAG"
echo "[run_training_subset] done: $RUN_DIR"
echo "[run_training_subset] verify loaded sizes:"
echo "  grep -E 'extracted_features_dataset|loaded [0-9]+ examples from: .*phones/train' \"$RUN_DIR/training.log\""
