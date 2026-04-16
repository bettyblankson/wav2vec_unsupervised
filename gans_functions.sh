#!/bin/bash

# This script runs the GANS training of  unsupervised wav2vec pipeline

# Wav2Vec Unsupervised Pipeline Runner
# This script runs the entire fairseq wav2vec unsupervised pipeline
# with checkpointing to allow resuming from any step

set -e  # Exit on error
set -o pipefail  # Exit if any command in a pipe fails

source utils.sh

#=========================== GANS training and preparation ==============================
train_gans(){
   local step_name="train_gans"
   export FAIRSEQ_ROOT=$FAIRSEQ_ROOT
   # export KALDI_ROOT="$DIR_PATH/pykaldi/tools/kaldi"
   export KENLM_ROOT="$KENLM_ROOT"
   export PYTHONPATH="$FAIRSEQ_ROOT:${PYTHONPATH:-}"
   local RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
   local RUN_DIR="$RESULTS_DIR/runs/$RUN_TAG"
   mkdir -p "$RUN_DIR"

   if is_completed "$step_name"; then
        log "Skipping gans training  (already completed)"
        return 0
    fi

    log "gans training."
    mark_in_progress "$step_name"
   

   # Set GAN_HPARAM_SWEEP=1 to reproduce multi-seed / multi-hparam grid (slow on CPU).
   if [ "${GAN_HPARAM_SWEEP:-0}" = "1" ]; then
     HPARAMS=(model.code_penalty=7,9 model.gradient_penalty=0.55,0.8 "common.seed=range(0,5)")
   else
     # Balanced defaults for 2k-data run: improve diversity without over-constraining.
     HPARAMS=(model.code_penalty=8.0 model.gradient_penalty=0.68 common.seed=0)
   fi

   # Upper bound on updates (utils.sh default); early stopping via checkpoint.patience.
   MAXU=()
   if [ -n "${GAN_MAX_UPDATE:-}" ]; then
     MAXU=(optimization.max_update="${GAN_MAX_UPDATE}")
   fi

   # Patience = number of validation rounds without improvement before stopping (validate every 1000 updates).
   EARLY=()
   if [ "${GAN_EARLY_STOP_PATIENCE:-15}" -gt 0 ] 2>/dev/null; then
     EARLY=(
       checkpoint.patience="${GAN_EARLY_STOP_PATIENCE:-15}"
       checkpoint.maximize_best_checkpoint_metric=false
     )
   fi

   PREFIX=w2v_unsup_gan_xp fairseq-hydra-train \
    -m --config-dir "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/gan" \
    --config-name w2vu \
    task.data="$CLUSTERING_DIR/precompute_pca512_cls128_mean_pooled" \
    task.text_data="$TEXT_OUTPUT/phones/" \
    task.kenlm_path="$TEXT_OUTPUT/phones/lm.phones.filtered.04.bin" \
    +task.vocab_usage_power=1.8 \
    common.user_dir="$FAIRSEQ_ROOT/examples/wav2vec/unsupervised" \
    model.smoothness_weight='1.6' \
    optimization.clip_norm=4.5 \
    dataset.batch_size="${NEW_BATCH_SIZE}" \
    "${MAXU[@]}" \
    "${EARLY[@]}" \
    "${HPARAMS[@]}" \
    +optimizer.groups.generator.optimizer.lr="[0.00003]" \
    +optimizer.groups.discriminator.optimizer.lr="[0.000014]" \
    ~optimizer.groups.generator.optimizer.amsgrad \
    ~optimizer.groups.discriminator.optimizer.amsgrad \
    2>&1 | tee "$RUN_DIR/training.log"

    

   if [ $? -eq 0 ]; then
        mark_completed "$step_name"
        log "gans trained successfully"
        # Promote newest checkpoint_best.pt from Hydra outputs to results (for inference / reports).
        best_ckpt=""
        if [ -d "$REPO_ROOT/multirun" ]; then
            best_ckpt=$(find "$REPO_ROOT/multirun" -name checkpoint_best.pt -type f -printf '%T@\t%p\n' 2>/dev/null | sort -n | tail -1 | cut -f2-)
        fi
        if [ -z "$best_ckpt" ] && [ -d "$REPO_ROOT/outputs" ]; then
            best_ckpt=$(find "$REPO_ROOT/outputs" -name checkpoint_best.pt -type f -printf '%T@\t%p\n' 2>/dev/null | sort -n | tail -1 | cut -f2-)
        fi
        if [ -n "$best_ckpt" ] && [ -f "$best_ckpt" ]; then
            cp -f "$best_ckpt" "$RUN_DIR/checkpoint_best_weighted_lm_ppl.pt"
            log "Copied best checkpoint to $RUN_DIR/checkpoint_best_weighted_lm_ppl.pt (use this model, not checkpoint_last.pt)"
        else
            log "NOTE: Could not auto-locate checkpoint_best.pt; search under $REPO_ROOT/multirun or outputs for the newest run."
        fi
        METRICS_SCRIPT="$REPO_ROOT/unsupervised_wav/scripts/plot_w2vu_training_metrics.py"
        if [ -f "$METRICS_SCRIPT" ] && [ -f "$RUN_DIR/training.log" ]; then
            log "Writing metric tables and graphs to $RUN_DIR/w2vu_metrics_report"
            python3 "$METRICS_SCRIPT" --log "$RUN_DIR/training.log" --out "$RUN_DIR/w2vu_metrics_report" \
                || log "WARNING: plot_w2vu_training_metrics.py failed (install python3-matplotlib for PNGs)"
        fi
    else
        log "ERROR: gans training failed"
        exit 1
    fi
}

