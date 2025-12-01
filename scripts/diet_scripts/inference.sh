#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/config.sh"

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Run DiET inference on distilled model"
    echo ""
    echo "Options:"
    echo "  --dataset       Dataset: mnist, xray, celeba (default: mnist)"
    echo "  --ups           Upsampling factor (default: 4)"
    echo "  --lr            Learning rate (default: 2000)"
    echo "  --model-path    Path to distilled model"
    echo "  --output-dir    Output directory"
    echo "  --gpu           GPU device ID (default: 0)"
    echo "  -h, --help      Show this help message"
}

DATASET="mnist"
UPS=4
LR=2000
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
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
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

DIET_DISTILL_DIR="${OUTPUTS_DIR}/diet/distillation"

if [ -z "${OUTPUT_DIR}" ]; then
    OUTPUT_DIR="${DIET_DISTILL_DIR}/${DATASET}_ups${UPS}_outdir"
fi

if [ -z "${MODEL_PATH}" ]; then
    MODEL_PATH="${OUTPUT_DIR}/fs_1.pth"
fi

export CUDA_VISIBLE_DEVICES="${GPU}"

log_info "Running DiET inference"
log_info "Dataset: ${DATASET}"
log_info "Upsampling: ${UPS}"
log_info "Model path: ${MODEL_PATH}"
log_info "Output dir: ${OUTPUT_DIR}"

case ${DATASET} in
    mnist)
        DATA_PATH="${DATA_DIR}/diet/hard_mnist/"
        ;;
    xray)
        DATA_PATH="${DATA_DIR}/diet/chest-xray/"
        ;;
    celeba)
        DATA_PATH="${DATA_DIR}/diet/celeba/"
        ;;
esac

python3 "${SCRIPT_DIR}/diet_scripts/python/inference.py" \
    --dataset "${DATASET}" \
    --data-dir "${DATA_PATH}" \
    --model-path "${MODEL_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --lr "${LR}" \
    --ups "${UPS}"

log_success "Inference completed"
