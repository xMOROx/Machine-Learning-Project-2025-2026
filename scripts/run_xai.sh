#!/bin/bash
# DiET vs Basic XAI Methods Comparison Framework
# Shell script wrapper for run_xai_experiments.py

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
XAI_DIR="${SCRIPT_DIR}/xai_experiments"

CONFIG_FILE="${SCRIPT_DIR}/config.sh"
if [ -f "${CONFIG_FILE}" ]; then
  source "${CONFIG_FILE}"
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

print_usage() {
  echo "Usage: $0 [OPTIONS]"
  echo ""
  echo "DiET vs Basic XAI Methods Comparison Framework"
  echo ""
  echo "This script compares DiET discriminative feature attribution with:"
  echo "  - GradCAM for image classification"
  echo "  - Integrated Gradients for text classification"
  echo ""
  echo "Datasets:"
  echo "  Images: CIFAR-10, CIFAR-100, SVHN, Fashion-MNIST"
  echo "  Text:   SST-2, IMDB, AG News"
  echo ""
  echo "Options:"
  echo "  --images         Run image comparison only (DiET vs GradCAM)"
  echo "  --text           Run text comparison only (DiET vs IG)"
  echo "  --top-k N        Number of top tokens to show (default: 5)"
  echo "  --low-vram       Use low VRAM settings for GPUs with < 8GB"
  echo "  --cpu            Force CPU mode (no GPU)"
  echo "  --gpu ID         GPU device ID (default: 0)"
  echo "  --epochs N       Number of training epochs"
  echo "  --batch-size N   Batch size"
  echo "  --skip-training  Skip training, use saved models"
  echo "  --data-dir DIR   Data directory (default: ${PROJECT_ROOT}/data)"
  echo "  --output-dir DIR Output directory (default: ${PROJECT_ROOT}/outputs/xai_experiments)"
  echo "  -h, --help       Show this help message"
  echo ""
  echo "Examples:"
  echo "  $0                              # Run full comparison (images + text)"
  echo "  $0 --images                     # DiET vs GradCAM on all image datasets"
  echo "  $0 --text --top-k 10            # DiET vs IG with top 10 tokens"
  echo "  $0 --low-vram                   # Low memory mode"
  echo "  $0 --skip-training              # Use saved models"
  echo ""
  echo "RTX 3060 Recommended Settings:"
  echo "  $0 --epochs 10 --batch-size 32  # Balanced speed/memory"
  echo "  $0 --low-vram                   # Memory optimization"
}

LOW_VRAM=""
CPU=""
GPU="0"
EPOCHS=""
BATCH_SIZE=""
SKIP_TRAINING=""
DIET_IMAGES=""
DIET_TEXT=""
TOP_K=""
DATA_DIR="${PROJECT_ROOT}/data"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/xai_experiments"

while [[ $# -gt 0 ]]; do
  case $1 in
  --images)
    DIET_IMAGES="--diet-images"
    shift
    ;;
  --text)
    DIET_TEXT="--diet-text"
    shift
    ;;
  --top-k)
    TOP_K="--top-k $2"
    shift 2
    ;;
  --low-vram)
    LOW_VRAM="--low-vram"
    shift
    ;;
  --cpu)
    CPU="--cpu"
    shift
    ;;
  --gpu)
    GPU="$2"
    shift 2
    ;;
  --epochs)
    EPOCHS="--epochs $2"
    shift 2
    ;;
  --batch-size)
    BATCH_SIZE="--batch-size $2"
    shift 2
    ;;
  --skip-training)
    SKIP_TRAINING="--skip-training"
    shift
    ;;
  --data-dir)
    DATA_DIR="$2"
    shift 2
    ;;
  --output-dir)
    OUTPUT_DIR="$2"
    shift 2
    ;;
  -h | --help)
    print_usage
    exit 0
    ;;
  *)
    log_error "Unknown option: $1"
    print_usage
    exit 1
    ;;
  esac
done

mkdir -p "${DATA_DIR}"
mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU}"

log_info "==================================================="
log_info "DiET vs Basic XAI Methods Comparison Framework"
log_info "==================================================="
log_info "Data directory: ${DATA_DIR}"
log_info "Output directory: ${OUTPUT_DIR}"
log_info "GPU: ${GPU}"
[ -n "${LOW_VRAM}" ] && log_info "Low VRAM mode: enabled"
[ -n "${CPU}" ] && log_info "CPU mode: enabled"
[ -n "${EPOCHS}" ] && log_info "Epochs: ${EPOCHS#--epochs }"
[ -n "${BATCH_SIZE}" ] && log_info "Batch size: ${BATCH_SIZE#--batch-size }"
[ -n "${TOP_K}" ] && log_info "Top-K tokens: ${TOP_K#--top-k }"
[ -n "${DIET_IMAGES}" ] && log_info "Mode: Images only (DiET vs GradCAM)"
[ -n "${DIET_TEXT}" ] && log_info "Mode: Text only (DiET vs IG)"
[ -z "${DIET_IMAGES}" ] && [ -z "${DIET_TEXT}" ] && log_info "Mode: Full comparison (Images + Text)"
log_info "==================================================="

cd "${PROJECT_ROOT}" || exit 1

log_info "Starting DiET comparison..."
python3 "${XAI_DIR}/run_xai_experiments.py" \
  --diet \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --gpu "${GPU}" \
  ${LOW_VRAM} \
  ${CPU} \
  ${EPOCHS} \
  ${BATCH_SIZE} \
  ${SKIP_TRAINING} \
  ${DIET_IMAGES} \
  ${DIET_TEXT} \
  ${TOP_K}

EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
  log_success "Comparison completed successfully!"
  log_info "Results saved to: ${OUTPUT_DIR}"
else
  log_error "Comparison failed with exit code: ${EXIT_CODE}"
fi

exit ${EXIT_CODE}
