#!/bin/bash
# 500 training-utterance preset: re-runs subsample → clustering → text → GAN into a dedicated results folder.
#
# Usage (from repo root, inside WSL, venv activated by child scripts):
#   chmod +x run_training_500utt.sh
#   ./run_training_500utt.sh TRAIN_WAV_DIR VAL_WAV_DIR TEST_WAV_DIR UNLABELLED_TEXT.txt
#
# Optional env (defaults shown):
#   MAX_UNLABELLED_TEXT_LINES=3000   # raw LM lines fed to prepare_text (~700+ phones/train typical; verify in log)
#   GAN_MAX_UPDATE=10000             # default = MAX_TRAIN_UTTERANCES*20; override if needed
#   RUN_TAG=run_500_train_audio_...  # archive folder under unsupervised_wav/data/results/librispeech/runs/
#   RESET_CHECKPOINTS=0              # skip sed cleanup if you already cleared progress.checkpoint manually
#
# Must run with bash (not `sh`): dash does not support `pipefail`. If you see "pipefail: invalid option",
# run: bash run_training_500utt.sh ...   or fix CRLF: sed -i 's/\r$//' run_training_500utt.sh
if [ -z "${BASH_VERSION:-}" ]; then
  echo "Use bash to run this script: bash $0 $*" >&2
  exit 1
fi
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 TRAIN_WAV_DIR VAL_WAV_DIR TEST_WAV_DIR UNLABELLED_TEXT_FILE"
  exit 1
fi

# --- Audio/text caps (500 train, 50 val; text sized to target a larger phones/train than the ~586 tiny run) ---
export MAX_TRAIN_UTTERANCES=500
export MAX_VALID_UTTERANCES=50
export MAX_UNLABELLED_TEXT_LINES="${MAX_UNLABELLED_TEXT_LINES:-3000}"

# Match run_training_subset: ~20 optimizer updates per training utterance (500 → 10k).
export GAN_MAX_UPDATE="${GAN_MAX_UPDATE:-$((MAX_TRAIN_UTTERANCES * 20))}"

export RUN_TAG="${RUN_TAG:-run_500_train_audio_$(date +%Y%m%d_%H%M)}"

CHECKPOINT_FILE="$REPO_ROOT/unsupervised_wav/data/checkpoints/librispeech/progress.checkpoint"
if [[ "${RESET_CHECKPOINTS:-1}" == "1" ]]; then
  mkdir -p "$(dirname "$CHECKPOINT_FILE")"
  [[ -f "$CHECKPOINT_FILE" ]] || touch "$CHECKPOINT_FILE"
  # Include manifest steps so train/valid.tsv are rebuilt from disk (not stuck at an old 200-utt cap).
  for step in create_manifests_train create_manifests_val create_manifests_test \
              subsample_manifests create_rVADfast remove_silence \
              create_manifests_nonsil_train create_manifests_nonsil_val \
              prepare_audio prepare_text train_gans; do
    sed -i "/^${step}:COMPLETED\$/d" "$CHECKPOINT_FILE" 2>/dev/null || true
    sed -i "/^${step}:IN_PROGRESS\$/d" "$CHECKPOINT_FILE" 2>/dev/null || true
  done
  echo "[run_training_500utt] Cleared checkpoint markers for: subsample → prepare_audio/text → train_gans"
fi

echo "[run_training_500utt] RUN_TAG=$RUN_TAG (GAN log + checkpoints + w2vu_metrics_report will live under runs/)"
echo "[run_training_500utt] After training, confirm data scale with: grep -E 'extracted_features_dataset|phones/train' unsupervised_wav/data/results/librispeech/runs/$RUN_TAG/training.log"

bash "$REPO_ROOT/run_wav2vec.sh" "$@"
bash "$REPO_ROOT/run_gans.sh"

echo "[run_training_500utt] Done. Outputs: unsupervised_wav/data/results/librispeech/runs/$RUN_TAG/"
