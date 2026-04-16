#!/bin/bash

# ==================== CONFIGURATION ====================
# Set these variables according to your environment and needs

# Main directories
# Keep everything self-contained under this repository.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIR_PATH="$REPO_ROOT/unsupervised_wav" # root directory used by scripts
DATA_ROOT="$DIR_PATH/data" # stores all the data generated from pipeline
FAIRSEQ_ROOT="$DIR_PATH/fairseq_" # root of the fairseq fork cloned during setup
KENLM_ROOT="$DIR_PATH/kenlm/build/bin"  # Path to KenLM installation
VENV_PATH="$DIR_PATH/venv"    # Path to virtual environment
RVAD_ROOT="$DIR_PATH/rVADfast/src/rVADfast" # root directory of rVADfast

GANS_OUTPUT_PHONES="$DATA_ROOT/transcription_phones"



# ==================== HELPER FUNCTIONS ====================

#fairseq file paths with slight changes made 
SPEECHPROCS="$DIR_PATH/rVADfast/src/rVADfast/speechproc/speechproc.py"
PREPARE_AUDIO="$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/prepare_audio.sh"
ADD_SELF_LOOP_SIMPLE="$FAIRSEQ_ROOT/examples/speech_recognition/kaldi/add-self-loop-simple.cc"
OPENFST_PATH="$DIR_PATH/fairseq/examples/speech_recognition/kaldi/kaldi_initializer.py"


# Arguments/variables
# For small test runs on limited RAM, use all available audio for clustering.
NEW_SAMPLE_PCT=1.0
# Use 1 for tiny text corpora (phone dict must not collapse to <SIL> only).
MIN_PHONES=1
# Batch size for GAN training (also patched into prepare_audio.sh). Use 8 if you hit OOM.
NEW_BATCH_SIZE=12

# Subset caps (0 = no limit). Applied after manifests are built.
#   MAX_TRAIN_UTTERANCES  → max rows in train.tsv (one training audio utterance / .wav per row).
#   MAX_VALID_UTTERANCES  → max rows in valid.tsv (validation audio utterances).
#   MAX_UNLABELLED_TEXT_LINES → first N lines of the unlabeled text file for prepare_text / LM.
# Example targets: 1000/150/4000 vs 2000/300/8000 (~4× text lines vs train utterances).
# NOTE: Fairseq logs "loaded N samples" from the *clustering/precompute* tree; if prepare_audio
# was not re-run after raising caps, N can stay small even when MAX_* is large (see TECHNICAL_REPORT §6).
# Override per run: export MAX_TRAIN_UTTERANCES=500 (etc.) before run_wav2vec.sh / run_gans.sh.
MAX_TRAIN_UTTERANCES="${MAX_TRAIN_UTTERANCES:-2000}"
MAX_VALID_UTTERANCES="${MAX_VALID_UTTERANCES:-300}"
MAX_UNLABELLED_TEXT_LINES="${MAX_UNLABELLED_TEXT_LINES:-8000}"

# Upper bound on GAN updates; early stopping usually finishes first.
#   export GAN_MAX_UPDATE=40000   # e.g. 2000 train utt × 20 (see run_training_subset.sh)
GAN_MAX_UPDATE="${GAN_MAX_UPDATE:-10000}"
export GAN_MAX_UPDATE
# Validation rounds without improvement (0 = disable).
#   export GAN_EARLY_STOP_PATIENCE=20
GAN_EARLY_STOP_PATIENCE="${GAN_EARLY_STOP_PATIENCE:-15}"
export GAN_EARLY_STOP_PATIENCE

PHONEMIZER="G2P"
LANG="en"

#models 
FASTTEXT_LIB_MODEL="$DIR_PATH/lid_model/lid.176.bin"  # the path to the language identification model
MODEL="$DIR_PATH/pre-trained/wav2vec_vox_new.pt" # the path to the pre-trained wav2vec model for audio feature extraction

# Dataset specifics
DATASET_NAME="librispeech"

# Output directories (will be created if they don't exist)
MANIFEST_DIR="$DATA_ROOT/manifests" # the directory that stores the manifest files for the audio dataset
NONSIL_AUDIO="$DATA_ROOT/processed_audio/" #the directory that stores the audio files with silence removed 
MANIFEST_NONSIL_DIR="$DATA_ROOT/manifests_nonsil" #the directory that stores the manifest files foe audio dataset with silence removed
CLUSTERING_DIR="$DATA_ROOT/clustering/$DATASET_NAME"  #stores the output of audio processing, the psuedophonemes(cluster IDs), Audio features
RESULTS_DIR="$DATA_ROOT/results/$DATASET_NAME" # Stores all the training information of the gans
CHECKPOINT_DIR="$DATA_ROOT/checkpoints/$DATASET_NAME" # stores the progress checkpoint file which keeps track of processes implemented 
LOG_DIR="$DATA_ROOT/logs/$DATASET_NAME" #stores the pipeline logs 
TEXT_OUTPUT="$DATA_ROOT/text" # stores the processes output from the prepared text function 


# Checkpoint file to track progress
CHECKPOINT_FILE="$CHECKPOINT_DIR/progress.checkpoint"


# Log message with timestamp
log() {
    local message="$1"
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$timestamp] $message" | tee -a "$LOG_DIR/pipeline.log"
}

# Check if a step has been completed
is_completed() {
    local step="$1"
    if [ -f "$CHECKPOINT_FILE" ]; then
        grep -q "^$step:COMPLETED$" "$CHECKPOINT_FILE" && return 0
    fi
    return 1
}

# Check if a step is in progress (for recovery after crash)
is_in_progress() {
    local step="$1"
    if [ -f "$CHECKPOINT_FILE" ]; then
        grep -q "^$step:IN_PROGRESS$" "$CHECKPOINT_FILE" && return 0
    fi
    return 1
}

# Mark a step as completed
mark_completed() {
    local step="$1"
    echo "$step:COMPLETED" >> "$CHECKPOINT_FILE"
    log "Marked step '$step' as completed"
}

# Mark a step as in progress
mark_in_progress() {
    local step="$1"
    # First remove any existing in-progress markers for this step
    if [ -f "$CHECKPOINT_FILE" ]; then
        sed -i "/^$step:IN_PROGRESS$/d" "$CHECKPOINT_FILE"
    fi
    echo "$step:IN_PROGRESS" >> "$CHECKPOINT_FILE"
    log "Marked step '$step' as in progress"
}

setup_path() {
    export HYDRA_FULL_ERROR=1
    # KALDI_ROOT is optional in this workflow; avoid unbound-variable errors under `set -u`.
    local kaldi_lib=""
    if [[ -n "${KALDI_ROOT:-}" ]]; then
        kaldi_lib="${KALDI_ROOT}/src/lib:"
    fi
    export LD_LIBRARY_PATH="${kaldi_lib}${KENLM_ROOT}/lib:${LD_LIBRARY_PATH:-}"
}


# Activate virtual environment if provided

activate_venv() {
    if [ -n "$VENV_PATH" ]; then
        log "Activating virtual environment at $VENV_PATH"
        source "$VENV_PATH/bin/activate"
    fi
}


# Create directories if they don't exist
create_dirs() {
    mkdir -p "$MANIFEST_DIR" "$CLUSTERING_DIR" "$MANIFEST_NONSIL_DIR" \
             "$RESULTS_DIR" "$CHECKPOINT_DIR" "$LOG_DIR" "$TEXT_OUTPUT" "$GANS_OUTPUT_PHONES"
}

# Fairseq wav2vec manifest: line 1 is dataset root; remaining lines are utterances.
subsample_fairseq_manifest() {
    local tsv=$1
    local max_utts=$2
    if [[ ! -f "$tsv" ]]; then
        log "subsample_fairseq_manifest: missing $tsv"
        return 1
    fi
    if [[ -z "$max_utts" || "${max_utts}" -le 0 ]]; then
        return 0
    fi
    local n_data
    n_data=$(($(wc -l < "$tsv") - 1))
    if [[ "$n_data" -le "$max_utts" ]]; then
        log "Manifest $(basename "$tsv") has ${n_data} utterances (<= cap ${max_utts}); no subsample."
        return 0
    fi
    # Avoid `tail | head` under `set -o pipefail`: tail often exits 141 (SIGPIPE) and aborts the script.
    local tmp
    tmp=$(mktemp)
    awk -v max="$max_utts" 'NR == 1 { print; next } NR - 1 <= max' "$tsv" > "$tmp"
    mv "$tmp" "$tsv"
    log "Subsampled $(basename "$tsv") to ${max_utts} utterances (was ${n_data})."
}




