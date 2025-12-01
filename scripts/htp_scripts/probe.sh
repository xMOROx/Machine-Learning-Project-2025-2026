#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/config.sh"

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Train probes (linear/MLP) for How-to-Probe"
    echo ""
    echo "Options:"
    echo "  --dataset       Dataset: imagenet, coco, voc (default: imagenet)"
    echo "  --backbone      Backbone: resnet50, bcosresnet50 (default: resnet50)"
    echo "  --probe-type    Probe type: linear, bcos-1, bcos-2, bcos-3, std-2, std-3 (default: linear)"
    echo "  --loss          Loss function: bce, ce (default: bce)"
    echo "  --checkpoint    Path to pretrained backbone checkpoint"
    echo "  --data-root     Path to dataset root"
    echo "  --output-dir    Output directory for probe checkpoints"
    echo "  --epochs        Number of epochs (default: 100)"
    echo "  --gpus          Number of GPUs (default: 4)"
    echo "  --resume        Resume from checkpoint"
    echo "  -h, --help      Show this help message"
}

DATASET="imagenet"
BACKBONE="resnet50"
PROBE_TYPE="linear"
LOSS="bce"
EPOCHS=100
GPUS=4
RESUME="auto"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --backbone)
            BACKBONE="$2"
            shift 2
            ;;
        --probe-type)
            PROBE_TYPE="$2"
            shift 2
            ;;
        --loss)
            LOSS="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --data-root)
            DATA_ROOT="$2"
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

if [ -z "${DATA_ROOT}" ]; then
    case ${DATASET} in
        imagenet)
            DATA_ROOT="${DATA_DIR}/htp/imagenet"
            ;;
        coco)
            DATA_ROOT="${DATA_DIR}/htp/coco"
            ;;
        voc)
            DATA_ROOT="${DATA_DIR}/htp/voc/VOCdevkit"
            ;;
    esac
fi

if [ -z "${OUTPUT_DIR}" ]; then
    OUTPUT_DIR="${OUTPUTS_DIR}/htp/probing/${DATASET}/${BACKBONE}_${PROBE_TYPE}_${LOSS}"
fi

ensure_dir "${OUTPUT_DIR}"

cd "${HTP_MMPRETRAIN_DIR}"
export PYTHONPATH="${HTP_MMPRETRAIN_DIR}:${PYTHONPATH}"

HTP_PROBING_DIR="${HTP_DIR}/probing"

case ${DATASET} in
    imagenet)
        CONFIG_DIR="${HTP_PROBING_DIR}/single_label_classification"
        ;;
    coco|voc)
        CONFIG_DIR="${HTP_PROBING_DIR}/multi_label_classification/${DATASET}"
        ;;
    *)
        log_error "Unknown dataset: ${DATASET}"
        exit 1
        ;;
esac

build_config_name() {
    local backbone=$1
    local probe=$2
    local loss=$3
    local dataset=$4
    
    if [ "${dataset}" = "imagenet" ]; then
        case ${probe} in
            linear)
                if [ "${backbone}" = "bcosresnet50" ]; then
                    echo "${backbone}_std-linear-postavgpool-no-bias_${loss}_4xb64-coslr-100e_in1k.py"
                else
                    echo "${backbone}_std-linear-postavgpool-with-bias_${loss}_4xb64-coslr-100e_in1k.py"
                fi
                ;;
            bcos-1|bcos-2|bcos-3)
                local layers="${probe#bcos-}"
                echo "${backbone}_bcos-linear-${layers}-postavgpool_${loss}_4xb64-linear-steplr-100e_in1k.py"
                ;;
            std-2|std-3)
                local layers="${probe#std-}"
                if [ "${backbone}" = "bcosresnet50" ]; then
                    echo "${backbone}_std-${layers}-linear-postavgpool-no-bias_${loss}_4xb64-linear-steplr-100e_in1k.py"
                else
                    echo "${backbone}_std-${layers}-linear-postavgpool-with-bias_${loss}_4xb64-linear-steplr-100e_in1k.py"
                fi
                ;;
        esac
    else
        case ${probe} in
            linear)
                echo "${backbone}frozen-multilabellinearclshead_4xb16_${dataset}14-448px.py"
                ;;
            bcos-1|bcos-2|bcos-3)
                local layers="${probe#bcos-}"
                echo "${backbone}frozen-multilabelbcos-${layers}-linearclshead_4xb16_${dataset}14-448px.py"
                ;;
            std-2|std-3)
                local layers="${probe#std-}"
                echo "${backbone}frozen-multilabelstd-${layers}-linearclshead_4xb16_${dataset}14-448px.py"
                ;;
        esac
    fi
}

CONFIG_NAME=$(build_config_name "${BACKBONE}" "${PROBE_TYPE}" "${LOSS}" "${DATASET}")
CONFIG_FILE="${CONFIG_DIR}/${CONFIG_NAME}"

if [ ! -f "${CONFIG_FILE}" ]; then
    log_error "Config file not found: ${CONFIG_FILE}"
    log_info "Available configs in ${CONFIG_DIR}:"
    ls "${CONFIG_DIR}"
    exit 1
fi

log_info "Training probe"
log_info "Dataset: ${DATASET}"
log_info "Backbone: ${BACKBONE}"
log_info "Probe type: ${PROBE_TYPE}"
log_info "Loss: ${LOSS}"
log_info "Config: ${CONFIG_FILE}"
log_info "Output: ${OUTPUT_DIR}"

GPU_IDS=$(seq -s, 0 $((GPUS-1)))

EXTRA_ARGS=""
if [ -n "${CHECKPOINT}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --cfg-options model.backbone.init_cfg.checkpoint=${CHECKPOINT}"
fi

CUDA_VISIBLE_DEVICES=${GPU_IDS} PORT=29500 \
    bash ./tools/dist_train.sh "${CONFIG_FILE}" "${GPUS}" \
    --resume "${RESUME}" \
    --work-dir "${OUTPUT_DIR}" \
    ${EXTRA_ARGS}

log_success "Probing training completed"
