#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/config.sh"

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Evaluate DiET models with pixel perturbation and IOU metrics"
    echo ""
    echo "Options:"
    echo "  --dataset       Dataset: mnist, xray, celeba (default: mnist)"
    echo "  --ups           Upsampling factor (default: 4)"
    echo "  --batch-size    Batch size (default: ${DEFAULT_EVAL_BATCH_SIZE})"
    echo "  --model-path    Path to model"
    echo "  --mask-path     Path to mask directory"
    echo "  --mask-num      Mask number (default: 1)"
    echo "  --perturbations Perturbation percentages (default: 10 20 50 100)"
    echo "  --eval-type     Evaluation type: perturbation, perturbation_distilled, iou, all (default: all)"
    echo "  --gpu           GPU device ID (default: 0)"
    echo "  --low-vram      Use low VRAM settings for small GPUs"
    echo "  -h, --help      Show this help message"
}

DATASET="mnist"
UPS=4
BATCH_SIZE="${DEFAULT_EVAL_BATCH_SIZE}"
MASK_NUM=1
PERTURBATIONS="10 20 50 100"
EVAL_TYPE="all"
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
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --mask-path)
            MASK_PATH="$2"
            shift 2
            ;;
        --mask-num)
            MASK_NUM="$2"
            shift 2
            ;;
        --perturbations)
            PERTURBATIONS="$2"
            shift 2
            ;;
        --eval-type)
            EVAL_TYPE="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --low-vram)
            BATCH_SIZE=8
            export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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
PYTHON_DIR="$(dirname "${BASH_SOURCE[0]}")/python"

if [ -z "${MODEL_PATH}" ]; then
    MODEL_PATH="${DIET_MODELS_DIR}/hard_${DATASET}_rn34.pth"
fi

if [ -z "${MASK_PATH}" ]; then
    MASK_PATH="${DIET_DISTILL_DIR}/${DATASET}_ups${UPS}_outdir"
fi

# Set data path based on dataset
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

export CUDA_VISIBLE_DEVICES="${GPU}"

run_perturbation() {
    log_info "Running pixel perturbation evaluation..."
    python3 "${PYTHON_DIR}/evaluate.py" \
        --eval-type perturbation \
        --dataset "${DATASET}" \
        --data-dir "${DATA_PATH}" \
        --model-path "${MODEL_PATH}" \
        --ups "${UPS}" \
        --mask-path "${MASK_PATH}" \
        --mask-num "${MASK_NUM}" \
        --batch-size "${BATCH_SIZE}" \
        --perturbations ${PERTURBATIONS}
}

run_perturbation_distilled() {
    log_info "Running pixel perturbation (distilled model) evaluation..."
    DISTILLED_MODEL="${MASK_PATH}/fs_${MASK_NUM}.pth"
    python3 "${PYTHON_DIR}/evaluate.py" \
        --eval-type perturbation \
        --dataset "${DATASET}" \
        --data-dir "${DATA_PATH}" \
        --model-path "${DISTILLED_MODEL}" \
        --ups "${UPS}" \
        --mask-path "${MASK_PATH}" \
        --mask-num "${MASK_NUM}" \
        --batch-size "${BATCH_SIZE}" \
        --perturbations ${PERTURBATIONS}
}

run_iou() {
    log_info "Running IOU evaluation..."
    python3 "${PYTHON_DIR}/evaluate.py" \
        --eval-type iou \
        --dataset "${DATASET}" \
        --data-dir "${DATA_PATH}" \
        --model-path "${MODEL_PATH}" \
        --ups "${UPS}" \
        --mask-path "${MASK_PATH}" \
        --batch-size "${BATCH_SIZE}" \
        --mask-num "${MASK_NUM}"
}

case ${EVAL_TYPE} in
    perturbation)
        run_perturbation
        ;;
    perturbation_distilled)
        run_perturbation_distilled
        ;;
    iou)
        run_iou
        ;;
    all)
        run_perturbation
        run_perturbation_distilled
        run_iou
        ;;
    *)
        log_error "Unknown eval type: ${EVAL_TYPE}"
        exit 1
        ;;
esac

log_success "Evaluation completed"
