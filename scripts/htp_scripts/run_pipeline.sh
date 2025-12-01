#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/config.sh"

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Run complete How-to-Probe training pipeline"
    echo ""
    echo "Options:"
    echo "  --method        SSL method: mocov2, byol, dino (default: dino)"
    echo "  --backbone      Backbone: resnet50, bcosresnet50 (default: resnet50)"
    echo "  --probe-types   Probe types to train (comma-separated, default: linear,bcos-3)"
    echo "  --datasets      Datasets for probing (comma-separated, default: imagenet)"
    echo "  --loss          Loss function: bce, ce (default: bce)"
    echo "  --data-path     Path to ImageNet train directory"
    echo "  --gpus          Number of GPUs (default: 4)"
    echo "  --skip-pretrain Skip pretraining"
    echo "  --skip-probing  Skip probing"
    echo "  --skip-eval     Skip evaluation"
    echo "  -h, --help      Show this help message"
}

METHOD="dino"
BACKBONE="resnet50"
PROBE_TYPES="linear,bcos-3"
DATASETS="imagenet"
LOSS="bce"
GPUS=4
SKIP_PRETRAIN=false
SKIP_PROBING=false
SKIP_EVAL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --method)
            METHOD="$2"
            shift 2
            ;;
        --backbone)
            BACKBONE="$2"
            shift 2
            ;;
        --probe-types)
            PROBE_TYPES="$2"
            shift 2
            ;;
        --datasets)
            DATASETS="$2"
            shift 2
            ;;
        --loss)
            LOSS="$2"
            shift 2
            ;;
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --skip-pretrain)
            SKIP_PRETRAIN=true
            shift
            ;;
        --skip-probing)
            SKIP_PROBING=true
            shift
            ;;
        --skip-eval)
            SKIP_EVAL=true
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

HTP_SCRIPTS="${SCRIPT_DIR}/htp_scripts"
PRETRAIN_OUTPUT="${OUTPUTS_DIR}/htp/pretraining/${METHOD}_${BACKBONE}"

log_info "============================================"
log_info "How-to-Probe Training Pipeline"
log_info "Method: ${METHOD}"
log_info "Backbone: ${BACKBONE}"
log_info "Probe types: ${PROBE_TYPES}"
log_info "Datasets: ${DATASETS}"
log_info "Loss: ${LOSS}"
log_info "GPUs: ${GPUS}"
log_info "============================================"

if ! $SKIP_PRETRAIN; then
    log_info ""
    log_info ">>> Step 1: Pretraining ${METHOD} with ${BACKBONE}..."
    
    PRETRAIN_ARGS="--method ${METHOD} --backbone ${BACKBONE} --gpus ${GPUS}"
    if [ -n "${DATA_PATH}" ]; then
        PRETRAIN_ARGS="${PRETRAIN_ARGS} --data-path ${DATA_PATH}"
    fi
    
    bash "${HTP_SCRIPTS}/pretrain.sh" ${PRETRAIN_ARGS}
    
    CHECKPOINT="${PRETRAIN_OUTPUT}/checkpoint.pth"
fi

if ! $SKIP_PROBING; then
    log_info ""
    log_info ">>> Step 2: Training probes..."
    
    IFS=',' read -ra PROBE_ARRAY <<< "${PROBE_TYPES}"
    IFS=',' read -ra DATASET_ARRAY <<< "${DATASETS}"
    
    for dataset in "${DATASET_ARRAY[@]}"; do
        for probe in "${PROBE_ARRAY[@]}"; do
            log_info ""
            log_info "Training ${probe} probe on ${dataset}..."
            
            PROBE_ARGS="--dataset ${dataset} --backbone ${BACKBONE} --probe-type ${probe} --loss ${LOSS} --gpus ${GPUS}"
            if [ -n "${CHECKPOINT}" ] && [ -f "${CHECKPOINT}" ]; then
                PROBE_ARGS="${PROBE_ARGS} --checkpoint ${CHECKPOINT}"
            fi
            
            bash "${HTP_SCRIPTS}/probe.sh" ${PROBE_ARGS}
        done
    done
fi

if ! $SKIP_EVAL; then
    log_info ""
    log_info ">>> Step 3: Evaluating attributions..."
    
    IFS=',' read -ra DATASET_ARRAY <<< "${DATASETS}"
    
    for dataset in "${DATASET_ARRAY[@]}"; do
        case ${dataset} in
            imagenet)
                log_info "Running GridPG evaluation on ImageNet..."
                bash "${HTP_SCRIPTS}/evaluate.sh" --eval-type gridpg --dataset imagenet
                ;;
            coco|voc)
                log_info "Running EPG evaluation on ${dataset}..."
                bash "${HTP_SCRIPTS}/evaluate.sh" --eval-type epg --dataset "${dataset}"
                ;;
        esac
    done
fi

log_info ""
log_info "============================================"
log_success "How-to-Probe pipeline completed"
log_info "============================================"
