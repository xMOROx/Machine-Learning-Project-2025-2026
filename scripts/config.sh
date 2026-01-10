#!/bin/bash
# Configuration for XAI Experiments (DiET vs GradCAM/IG Comparison Framework)

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Directories
DATA_DIR="${PROJECT_ROOT}/data"
OUTPUTS_DIR="${PROJECT_ROOT}/outputs"
XAI_OUTPUTS_DIR="${OUTPUTS_DIR}/xai_experiments"
CHECKPOINTS_DIR="${XAI_OUTPUTS_DIR}/checkpoints"

# Low VRAM mode for laptops
LOW_VRAM="${LOW_VRAM:-false}"

if [ "${LOW_VRAM}" = "true" ]; then
  DEFAULT_BATCH_SIZE=16
  DEFAULT_EVAL_BATCH_SIZE=8
else
  DEFAULT_BATCH_SIZE=64
  DEFAULT_EVAL_BATCH_SIZE=64
fi

# Colors for logging
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
  if ! command -v "$1" &>/dev/null; then
    log_error "$1 is required but not installed."
    return 1
  fi
  return 0
}
