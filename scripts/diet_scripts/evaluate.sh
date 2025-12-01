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
    echo "  --model-path    Path to model"
    echo "  --mask-path     Path to mask directory"
    echo "  --mask-num      Mask number (default: 1)"
    echo "  --perturbations Perturbation percentages (default: 10 20 50 100)"
    echo "  --eval-type     Evaluation type: perturbation, perturbation_ours, iou (default: all)"
    echo "  --gpu           GPU device ID (default: 0)"
    echo "  -h, --help      Show this help message"
}

DATASET="mnist"
UPS=4
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

export CUDA_VISIBLE_DEVICES="${GPU}"

run_perturbation() {
    log_info "Running pixel perturbation evaluation..."
    python3 "${PYTHON_DIR}/evaluate.py" \
        --eval-type perturbation \
        --dataset "${DATASET}" \
        --model-path "${MODEL_PATH}" \
        --ups "${UPS}" \
        --mask-path "${MASK_PATH}" \
        --mask-num "${MASK_NUM}" \
        --perturbations ${PERTURBATIONS}
}

run_perturbation_ours() {
    log_info "Running pixel perturbation (ours) evaluation..."
    DISTILLED_MODEL="${MASK_PATH}/fs_${MASK_NUM}.pth"
    python3 "${PYTHON_DIR}/evaluate.py" \
        --eval-type perturbation_ours \
        --dataset "${DATASET}" \
        --model-path "${DISTILLED_MODEL}" \
        --ups "${UPS}" \
        --mask-path "${MASK_PATH}" \
        --mask-num "${MASK_NUM}" \
        --perturbations ${PERTURBATIONS}
}

run_iou() {
    log_info "Running IOU evaluation..."
    python3 "${PYTHON_DIR}/evaluate.py" \
        --eval-type iou \
        --model-path "${MODEL_PATH}" \
        --ups "${UPS}" \
        --mask-path "${MASK_PATH}" \
        --mask-num "${MASK_NUM}"
}

case ${EVAL_TYPE} in
    perturbation)
        run_perturbation
        ;;
    perturbation_ours)
        run_perturbation_ours
        ;;
    iou)
        run_iou
        ;;
    all)
        run_perturbation
        run_perturbation_ours
        run_iou
        ;;
    *)
        log_error "Unknown eval type: ${EVAL_TYPE}"
        exit 1
        ;;
esac

log_success "Evaluation completed"
