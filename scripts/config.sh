#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATA_DIR="${PROJECT_ROOT}/data"
OUTPUTS_DIR="${PROJECT_ROOT}/outputs"
MODELS_DIR="${PROJECT_ROOT}/outputs/models"

DIET_DIR="${PROJECT_ROOT}/DiET"
HTP_DIR="${PROJECT_ROOT}/how-to-probe"
HTP_MMPRETRAIN_DIR="${HTP_DIR}/pretraining/mmpretrain"
HTP_DINO_DIR="${HTP_DIR}/pretraining/dino"

# VRAM-friendly settings for small laptop machines
# Set LOW_VRAM=true to use lower batch sizes and memory-efficient settings
LOW_VRAM="${LOW_VRAM:-false}"

# Default batch sizes (adjusted if LOW_VRAM=true)
if [ "${LOW_VRAM}" = "true" ]; then
    DEFAULT_BATCH_SIZE=16
    DEFAULT_EVAL_BATCH_SIZE=8
    DEFAULT_GPUS=1
    # Enable memory efficient options
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
else
    DEFAULT_BATCH_SIZE=64
    DEFAULT_EVAL_BATCH_SIZE=64
    DEFAULT_GPUS=4
fi

GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

log_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

ensure_dir() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
        log_info "Created directory: $1"
    fi
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "$1 is required but not installed."
        return 1
    fi
    return 0
}
