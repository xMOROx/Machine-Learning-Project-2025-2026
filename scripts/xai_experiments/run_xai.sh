#!/bin/bash
#
# XAI Experiments Orchestration Shell Script
#
# This script provides a convenient wrapper for running XAI experiments
# with appropriate settings for RTX 3060 and similar GPUs.
#
# Usage:
#   ./run_xai.sh all              # Run all experiments
#   ./run_xai.sh cifar10          # Run CIFAR-10 only
#   ./run_xai.sh glue             # Run GLUE SST-2 only  
#   ./run_xai.sh compare          # Run model comparison
#   ./run_xai.sh all --low-vram   # Low VRAM mode
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
XAI_DIR="${SCRIPT_DIR}"

# Source common config if available
CONFIG_FILE="${SCRIPT_DIR}/../config.sh"
if [ -f "${CONFIG_FILE}" ]; then
    source "${CONFIG_FILE}"
fi

# Colors for output
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
    echo "Usage: $0 <experiment> [OPTIONS]"
    echo ""
    echo "XAI Experiments Orchestration Script for Machine Learning Project"
    echo ""
    echo "Experiments:"
    echo "  all       Run all experiments (CIFAR-10, GLUE, comparison, DiET)"
    echo "  cifar10   Run CIFAR-10 with GradCAM"
    echo "  glue      Run GLUE SST-2 with BERT and Integrated Gradients"
    echo "  compare   Run model comparison (CNN, RF, LightGBM, SVM, LR)"
    echo "  diet      Run DiET vs basic XAI comparison (images + text)"
    echo ""
    echo "Options:"
    echo "  --low-vram       Use low VRAM settings for GPUs with < 8GB"
    echo "  --cpu            Force CPU mode (no GPU)"
    echo "  --gpu ID         GPU device ID (default: 0)"
    echo "  --epochs N       Number of training epochs"
    echo "  --batch-size N   Batch size"
    echo "  --model-type T   CNN model type: simple or resnet (default: resnet)"
    echo "  --skip-training  Skip training, use saved models"
    echo "  --diet-images    DiET for images only (with diet experiment)"
    echo "  --diet-text      DiET for text only (with diet experiment)"
    echo "  --data-dir DIR   Data directory (default: ${PROJECT_ROOT}/data)"
    echo "  --output-dir DIR Output directory (default: ${PROJECT_ROOT}/outputs/xai_experiments)"
    echo "  -h, --help       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 all                              # Run everything"
    echo "  $0 cifar10 --epochs 5               # CIFAR-10 with 5 epochs"
    echo "  $0 glue --low-vram                  # GLUE with low VRAM"
    echo "  $0 compare                          # Model comparison"
    echo "  $0 diet                             # DiET comparison"
    echo "  $0 diet --diet-images               # DiET for images only"
    echo "  $0 cifar10 --skip-training          # Use saved model"
    echo ""
    echo "RTX 3060 Recommended Settings:"
    echo "  $0 all --epochs 10 --batch-size 32  # Balanced speed/memory"
    echo "  $0 diet --low-vram                  # DiET with memory optimization"
}

# Default values
EXPERIMENT=""
LOW_VRAM=""
CPU=""
GPU="0"
EPOCHS=""
BATCH_SIZE=""
MODEL_TYPE="resnet"
SKIP_TRAINING=""
DIET_IMAGES=""
DIET_TEXT=""
DATA_DIR="${PROJECT_ROOT}/data"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/xai_experiments"

# Parse arguments
if [ $# -eq 0 ]; then
    print_usage
    exit 0
fi

EXPERIMENT="$1"
shift

while [[ $# -gt 0 ]]; do
    case $1 in
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
        --model-type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --skip-training)
            SKIP_TRAINING="--skip-training"
            shift
            ;;
        --diet-images)
            DIET_IMAGES="--diet-images"
            shift
            ;;
        --diet-text)
            DIET_TEXT="--diet-text"
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
        -h|--help)
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

# Validate experiment
case ${EXPERIMENT} in
    all|cifar10|glue|compare|diet)
        ;;
    -h|--help|help)
        print_usage
        exit 0
        ;;
    *)
        log_error "Unknown experiment: ${EXPERIMENT}"
        print_usage
        exit 1
        ;;
esac

# Build experiment flag
case ${EXPERIMENT} in
    all)
        EXPERIMENT_FLAG="--all"
        ;;
    cifar10)
        EXPERIMENT_FLAG="--cifar10"
        ;;
    glue)
        EXPERIMENT_FLAG="--glue"
        ;;
    compare)
        EXPERIMENT_FLAG="--compare"
        ;;
    diet)
        EXPERIMENT_FLAG="--diet"
        ;;
esac

# Create directories
mkdir -p "${DATA_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Set environment
export CUDA_VISIBLE_DEVICES="${GPU}"

# Print configuration
log_info "==================================================="
log_info "XAI Experiments Orchestration"
log_info "==================================================="
log_info "Experiment: ${EXPERIMENT}"
log_info "Data directory: ${DATA_DIR}"
log_info "Output directory: ${OUTPUT_DIR}"
log_info "GPU: ${GPU}"
[ -n "${LOW_VRAM}" ] && log_info "Low VRAM mode: enabled"
[ -n "${CPU}" ] && log_info "CPU mode: enabled"
[ -n "${EPOCHS}" ] && log_info "Epochs: ${EPOCHS#--epochs }"
[ -n "${BATCH_SIZE}" ] && log_info "Batch size: ${BATCH_SIZE#--batch-size }"
log_info "Model type: ${MODEL_TYPE}"
log_info "==================================================="

# Change to project directory
cd "${PROJECT_ROOT}"

# Run the experiment
log_info "Starting experiment..."
python3 "${XAI_DIR}/run_xai_experiments.py" \
    ${EXPERIMENT_FLAG} \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --model-type "${MODEL_TYPE}" \
    --gpu "${GPU}" \
    ${LOW_VRAM} \
    ${CPU} \
    ${EPOCHS} \
    ${BATCH_SIZE} \
    ${SKIP_TRAINING} \
    ${DIET_IMAGES} \
    ${DIET_TEXT}

EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
    log_success "Experiment completed successfully!"
    log_info "Results saved to: ${OUTPUT_DIR}"
else
    log_error "Experiment failed with exit code: ${EXIT_CODE}"
fi

exit ${EXIT_CODE}
