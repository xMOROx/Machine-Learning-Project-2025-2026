#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/config.sh"

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Run DiET distillation training"
    echo ""
    echo "Options:"
    echo "  --dataset       Dataset: mnist, xray, celeba (default: mnist)"
    echo "  --ups           Upsampling factor (default: 4)"
    echo "  --lr            Learning rate (default: 2000)"
    echo "  --batch-size    Batch size (default: ${DEFAULT_BATCH_SIZE})"
    echo "  --model-path    Path to baseline model"
    echo "  --output-dir    Output directory for distilled model"
    echo "  --gpu           GPU device ID (default: 0)"
    echo "  --low-vram      Use low VRAM settings for small GPUs"
    echo "  -h, --help      Show this help message"
}

DATASET="mnist"
UPS=4
LR=2000
BATCH_SIZE="${DEFAULT_BATCH_SIZE}"
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
        --batch-size)
            BATCH_SIZE="$2"
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
        --low-vram)
            BATCH_SIZE=16
            shift
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

DIET_MODELS_DIR="${OUTPUTS_DIR}/diet/trained_models"
DIET_DISTILL_DIR="${OUTPUTS_DIR}/diet/distillation"

if [ -z "${MODEL_PATH}" ]; then
    case ${DATASET} in
        mnist)
            MODEL_PATH="${DIET_MODELS_DIR}/hard_mnist_rn34.pth"
            ;;
        xray)
            MODEL_PATH="${DIET_MODELS_DIR}/xray_rn34.pth"
            ;;
        celeba)
            MODEL_PATH="${DIET_MODELS_DIR}/celeba_rn34.pth"
            ;;
    esac
fi

if [ -z "${OUTPUT_DIR}" ]; then
    OUTPUT_DIR="${DIET_DISTILL_DIR}/${DATASET}_ups${UPS}_outdir"
fi

ensure_dir "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU}"

log_info "Running DiET distillation"
log_info "Dataset: ${DATASET}"
log_info "Upsampling: ${UPS}"
log_info "Learning rate: ${LR}"
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
    *)
        log_error "Unknown dataset: ${DATASET}"
        print_usage
        exit 1
        ;;
esac

python3 "${SCRIPT_DIR}/diet_scripts/python/distillation.py" \
    --dataset "${DATASET}" \
    --data-dir "${DATA_PATH}" \
    --model-path "${MODEL_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --lr "${LR}" \
    --ups "${UPS}" \
    --batch-size "${BATCH_SIZE}"

log_success "Distillation completed. Output saved to ${OUTPUT_DIR}"
