#!/bin/bash

# This script holds all the functions and libraries needed a successful setup 
# that ensure a smooth running of the fairseq wav2vec unsupervised pipeline

set -e                       # Exit on error
set -o pipefail              # Exit if any command in a pipe fails
set -x                       # Print each command for debugging

# ==================== CONFIGURATION ====================
# Install everything relative to this repository, so the repo stays self-contained.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Main directories
INSTALL_ROOT="$REPO_ROOT/unsupervised_wav"
FAIRSEQ_ROOT="$INSTALL_ROOT/fairseq_"
KENLM_ROOT="$INSTALL_ROOT/kenlm"
VENV_PATH="$INSTALL_ROOT/venv"
RVADFAST_ROOT="$INSTALL_ROOT/rVADfast"
FLASHLIGHT_SEQ_ROOT="$INSTALL_ROOT/sequence"

# CUDA is optional (CPU-only installs are supported).
CUDA="${CUDA:-12.3}"


# ==================== HELPER FUNCTIONS ====================

# Log message with timestamp
log() {
    local message="$1"
    local timestamp
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$timestamp] $message"
}

# Check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

get_system_cuda_suffix() {
    if ! command -v nvcc >/dev/null 2>&1; then
        return 0
    fi
    local cuda_version
    cuda_version=$(nvcc --version | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
}

# Create home and log directory 
create_dirs() {
    mkdir -p "$INSTALL_ROOT"
    mkdir -p "$INSTALL_ROOT/logs"
}


# ==================== SETUP STEPS ====================
setup_venv() {
    log "Setting up Python virtual environment..."

    # Prefer Python 3.10 if available, otherwise 3.11, otherwise system python3.
    local pybin="python3"
    if command -v python3.10 >/dev/null 2>&1; then
        pybin="python3.10"
    elif command -v python3.11 >/dev/null 2>&1; then
        pybin="python3.11"
    fi

    # If an existing venv is on Python 3.12, rebuild it with 3.10/3.11 because
    # this fairseq fork frequently fails to compile extensions on 3.12.
    if [ -d "$VENV_PATH" ]; then
        if [ -x "$VENV_PATH/bin/python" ] && "$VENV_PATH/bin/python" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3,12) else 1)' >/dev/null 2>&1; then
            log "[WARN] Existing venv uses Python 3.12. Recreating venv with $pybin..."
            rm -rf "$VENV_PATH"
        else
            log "Virtual environment already exists at $VENV_PATH"
        fi
    fi

    if [ ! -d "$VENV_PATH" ]; then
        "$pybin" -m venv --clear "$VENV_PATH"
        log "Created virtual environment at $VENV_PATH (using $pybin)"
    fi
    
    # Activate virtual environment
    source "$VENV_PATH/bin/activate"

    # Ensure packaging tooling exists in the venv.
    # Some distros/venv setups may not include setuptools by default.
    python -m pip install --upgrade "pip==24.0" setuptools wheel

    log "Python virtual environment setup completed."
}

#installing_python_basic_dependencies
basic_dependencies(){
    sudo apt-get update
    # Install Python toolchains. fairseq builds most reliably on Python 3.10/3.11.
    sudo apt-get install -y python3 python3-pip build-essential
    sudo apt-get install -y python3.10 python3.10-venv python3.10-dev || true
    sudo apt-get install -y python3.11 python3.11-venv python3.11-dev || true
    sudo apt-get install -y pciutils
    sudo apt-get install -y zsh
    sudo apt-get install autoconf automake cmake curl g++ git graphviz libatlas3-base libtool make pkg-config subversion unzip wget zlib1g-dev gfortran
    sudo apt update
    sudo apt install -y build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev wget curl
    sudo apt install software-properties-common

}

# This function installs the cuda version suited for your machine

cuda_installation() {
    local cmd_file="cuda_installation.txt"

    if [[ -f "$cmd_file" ]]; then
        echo "Starting installation from $cmd_file..."

        source "$cmd_file"

        # Add CUDA to PATH safely (without expanding PATH immediately)
        echo 'export PATH=/usr/local/cuda-'"$CUDA"'/bin:$PATH' >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH=/usr/local/cuda-'"$CUDA"'/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

        # Apply immediately to current shell
        export PATH="/usr/local/cuda-$CUDA/bin:$PATH"
        export LD_LIBRARY_PATH="/usr/local/cuda-$CUDA/lib64:$LD_LIBRARY_PATH"

        source ~/.bashrc

        echo "CUDA environment variables configured."
    else
        echo "Error: $cmd_file not found!"
        return 1
    fi
}


#Installation of gpu drivers and toolkit
gpu_drivers_installation(){
 echo "--- Starting GPU Driver and Toolkit Installation ---"
    # 1. Download Google Cloud GPU installation script
    echo "1. Downloading GCP GPU driver installation script..."
    curl -s -O https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py

    # 2. Run the installation script
    echo "2. Running GPU driver installation script"
    sudo python3 install_gpu_driver.py

    # 3. Update package lists
    echo "3. Updating package lists..."
    sudo apt-get update -y

    # --- 4. Verify nvidia-smi and Fix PATH if necessary ---
    echo "4. Verifying installation location and fixing PATH..."

    if command -v nvidia-smi >/dev/null 2>&1; then
        echo ""
        echo "=================================================================="
        echo "SUCCESS: 'nvidia-smi' is now found and running the command:"
        nvidia-smi
        echo "=================================================================="
    else
        echo ""
        echo "=================================================================="
        echo "FAILURE: 'nvidia-smi' is not found in the initial system PATH."
        
        if command -v nvidia-smi >/dev/null 2>&1; then
            echo "SUCCESS: PATH fix worked! Running the command:"
            nvidia-smi
            echo "=================================================================="
        else
            echo "CRITICAL FAILURE: Driver utilities are missing or severely misconfigured."
            echo "A system reboot may be required to fully activate the newly installed driver."
            echo "=================================================================="
        fi
    fi
}

# Install PyTorch and related packages
install_pytorch_and_other_packages() {
    log "Installing PyTorch and related packages..."
    source "$VENV_PATH/bin/activate"
  
    python -m pip install --upgrade pip

    # fairseq (this fork) is not compatible with NumPy 2.x C-API in our build path.
    # Force a NumPy 1.x version early so compiled extensions build reliably.
    pip install --upgrade --force-reinstall "numpy==1.26.4"

    # If torch is already installed (correct version), skip reinstall to avoid large downloads.
    if python -c "import torch; import sys; sys.exit(0 if torch.__version__.startswith('2.3.0') else 1)" >/dev/null 2>&1; then
        log "[INFO] torch==2.3.0 already installed. Skipping PyTorch install."
    else
    # Prefer CPU wheels unless a CUDA toolchain is present.
    if command -v nvcc >/dev/null 2>&1; then
        log "[INFO] nvcc detected. Installing CUDA-enabled PyTorch wheels."
        pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url "https://download.pytorch.org/whl/cu121"
    else
        log "[INFO] nvcc not found. Installing CPU-only PyTorch wheels."
        pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url "https://download.pytorch.org/whl/cpu"
    fi
    fi

    # Install other required packages
    pip install scipy tqdm sentencepiece soundfile librosa editdistance tensorboardX packaging soundfile
    pip install npy-append-array h5py kaldi-io g2p_en

    if ! command -v nvcc >/dev/null 2>&1; then
         pip install faiss-cpu
    else
        pip install faiss-gpu
    fi
    
    pip install ninja torchcodec

    log "PyTorch and related packages installed successfully."
}

# Clone and install fairseq
install_fairseq() {
    log "--- Installing fairseq ---"
    log "Activating virtual environment: $VENV_PATH"
    source "$VENV_PATH/bin/activate"
     pip install "pip==24.0"
    # source "$VENV_PATH/bin/activate"

    cd "$INSTALL_ROOT"

    if [ -d "$FAIRSEQ_ROOT" ]; then
        log "fairseq repository already exists. Pulling latest changes..."
        cd "$FAIRSEQ_ROOT"
        git pull || { log "[WARN] Failed to pull latest fairseq changes. Continuing with existing version."; }
    else
        log "Cloning fairseq repository..."
        # git clone https://github.com/facebookresearch/fairseq.git "$FAIRSEQ_ROOT" \
        git clone https://github.com/Ashesi-Org/fairseq_.git "$FAIRSEQ_ROOT" || { log "[ERROR] Failed to clone fairseq repository."; exit 1; }
        cd "$FAIRSEQ_ROOT"
    fi

    # PROSIT 3 / fork-specific changes live in a patch next to this repo's fairseq checkout
    # (fairseq_ is its own git repo, so the outer repo cannot track its files directly).
    local fairseq_patch="$INSTALL_ROOT/fairseq_soc.patch"
    if [ -f "$fairseq_patch" ]; then
        log "Applying fork patch: $fairseq_patch"
        if git apply --check "$fairseq_patch" 2>/dev/null; then
            git apply "$fairseq_patch" || { log "[ERROR] Failed to apply $fairseq_patch"; exit 1; }
        else
            log "[WARN] Patch does not apply cleanly (maybe already applied or fairseq revision drifted). Skipping apply. Inspect with: git apply --check $fairseq_patch"
        fi
    else
        log "[INFO] No fairseq patch at $fairseq_patch (optional)."
    fi

    log "Installing fairseq in editable mode (CPU-friendly)..."
    # Avoid pip build isolation: otherwise pip may try to download a CUDA-enabled torch stack
    # as a build dependency (very large) even though we already installed CPU torch.
    # Also avoid dependency resolution during editable install; we install requirements explicitly below.
    # Limit build parallelism to reduce RAM spikes in small machines.
    export MAX_JOBS="${MAX_JOBS:-1}"
    # Skip compiled extensions for Python 3.12 / low-resource environments.
    FAIRSEQ_SKIP_EXTENSIONS=1 PIP_NO_BUILD_ISOLATION=1 pip install --no-build-isolation --no-deps --editable ./ \
        || { log "[ERROR] Failed to install fairseq in editable mode."; exit 1; }

    # Because we install fairseq with --no-deps (to avoid large/unwanted torch/cuda dependency resolution),
    # we must explicitly install the small runtime dependencies that fairseq expects.
    pip install "omegaconf<2.1" "hydra-core>=1.0.7,<1.1" regex sacrebleu tqdm bitarray "scikit-learn" cffi cython packaging \
        || { log "[ERROR] Failed to install fairseq runtime dependencies."; exit 1; }

    # Install wav2vec specific requirements if the file exists
    local wav2vec_req_file="$FAIRSEQ_ROOT/examples/wav2vec/requirements.txt"
    if [ -f "$wav2vec_req_file" ]; then
        log "Installing wav2vec specific requirements from $wav2vec_req_file..."
        pip install -r "$wav2vec_req_file" \
            || { log "[WARN] Failed to install some wav2vec requirements. Check $wav2vec_req_file."; }
    else
        log "[INFO] No specific requirements file found at $wav2vec_req_file."
    fi

    log "fairseq installed successfully."
    deactivate
}


#Install rVADfast for audio silence removal
install_rVADfast() {
    log "Cloning and installing rVADfast..."
    cd "$INSTALL_ROOT"
    
    source "$VENV_PATH/bin/activate"

    if [ -d "$RVADFAST_ROOT" ]; then
        log "rVADfast already exists. Updating..."
        cd "$RVADFAST_ROOT"
        git pull
    else
        log "Cloning rVADfast repository..."
        git clone https://github.com/zhenghuatan/rVADfast.git "$RVADFAST_ROOT"
        cd "$RVADFAST_ROOT"
    fi

    mkdir -p "$RVADFAST_ROOT/src"
    
    log "rVADfast installed successfully."
}

#  Clone and build KenLM
install_kenlm() {
    log "Cloning and building KenLM..."
    cd "$INSTALL_ROOT"

    sudo apt update
    sudo apt install libeigen3-dev

    sudo apt update
    sudo apt install libboost-all-dev

    if [ -d "$KENLM_ROOT" ]; then
        log "KenLM repository already exists."
    else
        log "Cloning KenLM repository..."
        git clone https://github.com/kpu/kenlm.git "$KENLM_ROOT"
    fi
    
    cd "$KENLM_ROOT"
    if [ -d "build" ]; then
        log "KenLM build directory already exists. Skipping build step."
    else  
        mkdir -p build
        cd build
        cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        make -j $(nproc)
    fi
    
    source "$VENV_PATH/bin/activate"
    pip install https://github.com/kpu/kenlm/archive/master.zip
    
    log "KenLM built successfully."
}

#  Install Flashlight and Flashlight-Sequence
install_flashlight() {
    log "--- Installing Flashlight (Text and Sequence) ---"
    cd "$INSTALL_ROOT"

    sudo apt-get install pybind11-dev

    log "Activating virtual environment: $VENV_PATH"
    source "$VENV_PATH/bin/activate"

    # Install flashlight-text (Python-only package)
    log "Installing flashlight-text Python package..."
    pip install flashlight-text \
        || { log "[ERROR] Failed to install flashlight-text."; exit 1; }

    # Clone or update the sequence repository
    if [ -d "$FLASHLIGHT_SEQ_ROOT" ]; then
        log "Flashlight sequence repository already exists. Updating..."
        cd "$FLASHLIGHT_SEQ_ROOT"
        git pull || { log "[WARN] Failed to pull latest flashlight sequence changes."; }
    else
        log "Cloning flashlight sequence repository..."
        git clone https://github.com/flashlight/sequence.git "$FLASHLIGHT_SEQ_ROOT" \
            || { log "[ERROR] Failed to clone flashlight sequence."; exit 1; }
        cd "$FLASHLIGHT_SEQ_ROOT"
    fi

    log "Configuring and building flashlight sequence WITH Python bindings..."
    rm -rf build
    mkdir -p build && cd build

    local flashlight_python_flag="-DFLASHLIGHT_BUILD_PYTHON=ON"
    local use_cuda_flag="-DFLASHLIGHT_USE_CUDA=OFF"

    if command -v nvcc >/dev/null 2>&1; then
        log "[INFO] nvcc detected. Enabling CUDA build for flashlight sequence."
        use_cuda_flag="-DFLASHLIGHT_USE_CUDA=ON"
        export USE_CUDA=1
    else
        log "[INFO] nvcc not found. Building flashlight sequence CPU-only."
        export USE_CUDA=0
    fi

    local python_executable="$VENV_PATH/bin/python"
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DPYTHON_EXECUTABLE="$python_executable" \
             "$flashlight_python_flag" \
             "$use_cuda_flag"

    log "Building Flashlight sequence (C++ and Python)..."
    cmake --build . --config Release --parallel "$(nproc)"

    log "Installing Flashlight sequence Python bindings into venv..."
    cd ..
    pip install .

    log "Flashlight installation steps completed."

    # --- Re-install fairseq AFTER Flashlight bindings are in venv ---
    log "Re-installing fairseq to ensure it picks up Flashlight bindings..."
    install_fairseq # Call the fairseq install function again (it will activate/deactivate venv)

    log "--- Flashlight Installation Finished ---"
    # Final deactivate handled by install_fairseq
}

#  Download pre-trained wav2vec model
download_pretrained_model() {
    log "Downloading pre-trained wav2vec model..."
    
    mkdir -p "$INSTALL_ROOT/pre-trained"
    cd "$INSTALL_ROOT/pre-trained"
    
    if [ -f "$INSTALL_ROOT/pre-trained/wav2vec_vox_new.pt" ]; then
        log "Pre-trained model already exists. Skipping download."
    else
        wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_new.pt
    fi
    
    log "Pre-trained model downloaded successfully."
}

# Download language identification model
download_languageIdentification_model() {
    log "Downloading language identification model..."
    
    mkdir -p "$INSTALL_ROOT/lid_model"
    cd "$INSTALL_ROOT/lid_model"
    
    if [ -f "$INSTALL_ROOT/lid_model/lid.176.bin" ]; then
        log "LID model already exists. Skipping download."
    else
        wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
    fi

    source "$VENV_PATH/bin/activate"
    pip install fasttext
    
    log "Language identification model downloaded successfully."
}
