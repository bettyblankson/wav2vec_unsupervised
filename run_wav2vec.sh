#!/bin/bash

# Pass through dataset paths as $1–$4 (see wav2vec_functions.sh).
# Example: ./run_wav2vec.sh /path/to/train_wav /path/to/val_wav /path/to/test_wav /path/to/text.txt
# Source the function definitions
source "$(dirname "$0")/wav2vec_functions.sh" "$@"

create_dirs #creates directories for storing outputs from the different steps 

activate_venv  
setup_path  #add kenlm and kaldi to the LD_LIBRARY directory
    
log "Starting wav2vec unsupervised pipeline for $DATASET_NAME"
 
log "It creates a manifest files for the audio dataset audio format"

create_manifests_train 0 
create_manifests_val 0 
subsample_audio_manifests
create_manifests_test 0 

#creates new manifest with silence removed

create_rVADfast # identifies the sequence of silence in an audio 
remove_silence # removes the silence sequence found by rvad in the audio
create_manifests_nonsil_train 0.1
create_manifests_nonsil_val 0.1


# Train GANS: 
#     prepare_audio:  processes the unlabelled audio 
#     prepare_text: 

prepare_audio 
prepare_text  

log "Pipeline completed successfully!"
