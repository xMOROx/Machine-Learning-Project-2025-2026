#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/config.sh"

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Run complete DiET training pipeline"
    echo ""
    echo "Options:"
    echo "  --dataset       Dataset: mnist, xray, celeba (default: mnist)"
    echo "  --ups           Upsampling factor (default: 4)"
    echo "  --lr            Learning rate for distillation (default: 2000)"
    echo "  --epochs        Epochs for baseline training (default: 10)"
    echo "  --skip-baseline Skip baseline training"
    echo "  --skip-distill  Skip distillation"
    echo "  --skip-eval     Skip evaluation"
    echo "  --gpu           GPU device ID (default: 0)"
    echo "  -h, --help      Show this help message"
}

DATASET="mnist"
UPS=4
LR=2000
EPOCHS=10
SKIP_BASELINE=false
SKIP_DISTILL=false
SKIP_EVAL=false
GPU=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --ups)
            UPS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --skip-baseline)
            SKIP_BASELINE=true
            shift
            ;;
        --skip-distill)
            SKIP_DISTILL=true
            shift
            ;;
        --skip-eval)
            SKIP_EVAL=true
            shift
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

DIET_SCRIPTS="${SCRIPT_DIR}/diet_scripts"

log_info "============================================"
log_info "DiET Training Pipeline"
log_info "Dataset: ${DATASET}"
log_info "Upsampling: ${UPS}"
log_info "GPU: ${GPU}"
log_info "============================================"

if [ "${DATASET}" = "mnist" ] && [ ! -d "${DATA_DIR}/diet/hard_mnist/training/0" ]; then
    log_info ""
    log_info ">>> Step 0: Preparing Hard MNIST dataset..."
    bash "${DIET_SCRIPTS}/prepare_hard_mnist.sh"
fi

if ! $SKIP_BASELINE; then
    log_info ""
    log_info ">>> Step 1: Training baseline model..."
    bash "${DIET_SCRIPTS}/train_baseline.sh" \
        --dataset "${DATASET}" \
        --epochs "${EPOCHS}" \
        --gpu "${GPU}"
fi

if ! $SKIP_DISTILL; then
    log_info ""
    log_info ">>> Step 2: Running distillation..."
    bash "${DIET_SCRIPTS}/distillation.sh" \
        --dataset "${DATASET}" \
        --ups "${UPS}" \
        --lr "${LR}" \
        --gpu "${GPU}"
    
    log_info ""
    log_info ">>> Step 3: Running inference..."
    bash "${DIET_SCRIPTS}/inference.sh" \
        --dataset "${DATASET}" \
        --ups "${UPS}" \
        --lr "${LR}" \
        --gpu "${GPU}"
fi

if ! $SKIP_EVAL; then
    log_info ""
    log_info ">>> Step 4: Running evaluation..."
    bash "${DIET_SCRIPTS}/evaluate.sh" \
        --dataset "${DATASET}" \
        --ups "${UPS}" \
        --gpu "${GPU}"
fi

log_info ""
log_info "============================================"
log_success "DiET pipeline completed"
log_info "============================================"
