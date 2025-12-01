#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/config.sh"

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Train SSL models (MoCov2, BYOL, DINO) for How-to-Probe"
    echo ""
    echo "Options:"
    echo "  --method        SSL method: mocov2, byol, dino (required)"
    echo "  --backbone      Backbone: resnet50, bcosresnet50 (default: resnet50)"
    echo "  --data-path     Path to ImageNet train directory"
    echo "  --output-dir    Output directory for checkpoints"
    echo "  --epochs        Number of epochs (default: 200)"
    echo "  --batch-size    Batch size per GPU (default: 64)"
    echo "  --lr            Learning rate"
    echo "  --gpus          Number of GPUs (default: 4)"
    echo "  --resume        Resume from checkpoint (auto or path)"
    echo "  -h, --help      Show this help message"
}

METHOD=""
BACKBONE="resnet50"
EPOCHS=200
BATCH_SIZE=64
GPUS=4
RESUME="auto"
LR=""

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
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
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
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
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

if [ -z "${METHOD}" ]; then
    log_error "Method is required. Use --method mocov2|byol|dino"
    exit 1
fi

if [ -z "${DATA_PATH}" ]; then
    DATA_PATH="${DATA_DIR}/htp/imagenet/train"
fi

if [ -z "${OUTPUT_DIR}" ]; then
    OUTPUT_DIR="${OUTPUTS_DIR}/htp/pretraining/${METHOD}_${BACKBONE}"
fi

ensure_dir "${OUTPUT_DIR}"

log_info "Training SSL model"
log_info "Method: ${METHOD}"
log_info "Backbone: ${BACKBONE}"
log_info "Data path: ${DATA_PATH}"
log_info "Output dir: ${OUTPUT_DIR}"
log_info "GPUs: ${GPUS}"

case ${METHOD} in
    mocov2|byol)
        cd "${HTP_MMPRETRAIN_DIR}"
        export PYTHONPATH="${HTP_MMPRETRAIN_DIR}:${PYTHONPATH}"
        
        CONFIG_FILE="configs/${METHOD}/${METHOD}_${BACKBONE}_4xb64-coslr-${EPOCHS}e_in1k.py"
        
        if [ ! -f "${CONFIG_FILE}" ]; then
            log_error "Config file not found: ${CONFIG_FILE}"
            exit 1
        fi
        
        GPU_IDS=$(seq -s, 0 $((GPUS-1)))
        
        log_info "Running distributed training with ${GPUS} GPUs..."
        CUDA_VISIBLE_DEVICES=${GPU_IDS} PORT=29500 \
            bash ./tools/dist_train.sh "${CONFIG_FILE}" "${GPUS}" \
            --resume "${RESUME}" \
            --work-dir "${OUTPUT_DIR}"
        ;;
    
    dino)
        cd "${HTP_DINO_DIR}"
        export PYTHONPATH="${HTP_DINO_DIR}:${PYTHONPATH}"
        
        if [ "${BACKBONE}" = "bcosresnet50" ]; then
            if [ -z "${LR}" ]; then LR=0.003; fi
            EXTRA_ARGS="--weight_decay 0.0 --weight_decay_end 0.0 --optimizer adamw --use_bcos_head 1"
        else
            if [ -z "${LR}" ]; then LR=0.03; fi
            EXTRA_ARGS="--weight_decay 1e-4 --weight_decay_end 1e-4 --optimizer sgd"
        fi
        
        log_info "Running DINO training with ${GPUS} GPUs..."
        torchrun --nproc_per_node=${GPUS} main_dino.py \
            --arch "${BACKBONE}" \
            --lr "${LR}" \
            --data_path "${DATA_PATH}" \
            --output_dir "${OUTPUT_DIR}" \
            --epochs "${EPOCHS}" \
            --batch_size_per_gpu "${BATCH_SIZE}" \
            --global_crops_scale 0.14 1 \
            --local_crops_scale 0.05 0.14 \
            --warmup_teacher_temp_epochs 30 \
            --local_crops_number 8 \
            --global_crop_size 224 \
            --local_crop_size 96 \
            ${EXTRA_ARGS}
        ;;
    
    *)
        log_error "Unknown method: ${METHOD}"
        exit 1
        ;;
esac

log_success "Pretraining completed"
