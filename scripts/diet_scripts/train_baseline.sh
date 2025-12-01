#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/config.sh"

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Train baseline models for DiET"
    echo ""
    echo "Options:"
    echo "  --dataset     Dataset to train on: mnist, xray, celeba (default: mnist)"
    echo "  --epochs      Number of training epochs (default: 10)"
    echo "  --batch-size  Batch size (default: 64)"
    echo "  --lr          Learning rate (default: 0.001)"
    echo "  --data-dir    Custom data directory"
    echo "  --output-dir  Custom output directory"
    echo "  --gpu         GPU device ID (default: 0)"
    echo "  -h, --help    Show this help message"
}

DATASET="mnist"
EPOCHS=10
BATCH_SIZE=64
LR=0.001
GPU=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUTS_DIR="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
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

DIET_DATA_DIR="${DATA_DIR}/diet"
DIET_MODELS_DIR="${OUTPUTS_DIR}/diet/trained_models"
ensure_dir "${DIET_MODELS_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU}"

log_info "Training baseline model for DiET"
log_info "Dataset: ${DATASET}"
log_info "Epochs: ${EPOCHS}"
log_info "Batch size: ${BATCH_SIZE}"
log_info "Learning rate: ${LR}"
log_info "GPU: ${GPU}"

cd "${PROJECT_ROOT}"

case ${DATASET} in
    mnist)
        DATA_PATH="${DIET_DATA_DIR}/hard_mnist/"
        MODEL_OUT="${DIET_MODELS_DIR}/hard_mnist_rn34.pth"
        ;;
    xray)
        DATA_PATH="${DIET_DATA_DIR}/chest-xray/"
        MODEL_OUT="${DIET_MODELS_DIR}/xray_rn34.pth"
        ;;
    celeba)
        DATA_PATH="${DIET_DATA_DIR}/celeba/"
        MODEL_OUT="${DIET_MODELS_DIR}/celeba_rn34.pth"
        ;;
    *)
        log_error "Unknown dataset: ${DATASET}"
        exit 1
        ;;
esac

python3 "${SCRIPT_DIR}/diet_scripts/python/train_baseline.py" \
    --dataset "${DATASET}" \
    --data-dir "${DATA_PATH}" \
    --output-path "${MODEL_OUT}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}"

log_success "Training completed. Model saved to ${MODEL_OUT}"
