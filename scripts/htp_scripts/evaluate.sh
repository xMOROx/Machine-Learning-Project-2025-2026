#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/config.sh"

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Evaluate model attributions using GridPG or EPG metrics"
    echo ""
    echo "Options:"
    echo "  --eval-type     Evaluation type: gridpg, epg (required)"
    echo "  --dataset       Dataset: imagenet, coco, voc (default: imagenet for gridpg, voc for epg)"
    echo "  --model-path    Path to model checkpoint"
    echo "  --data-root     Path to dataset root"
    echo "  --output-dir    Output directory for results"
    echo "  --grid-size     Grid size for GridPG: 2 or 3 (default: 3)"
    echo "  --confidence    Confidence threshold for GridPG (default: 0.95)"
    echo "  -h, --help      Show this help message"
}

EVAL_TYPE=""
DATASET=""
GRID_SIZE=3
CONFIDENCE=0.95

while [[ $# -gt 0 ]]; do
    case $1 in
        --eval-type)
            EVAL_TYPE="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
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
        --grid-size)
            GRID_SIZE="$2"
            shift 2
            ;;
        --confidence)
            CONFIDENCE="$2"
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

if [ -z "${EVAL_TYPE}" ]; then
    log_error "Evaluation type is required. Use --eval-type gridpg|epg"
    exit 1
fi

if [ -z "${DATASET}" ]; then
    case ${EVAL_TYPE} in
        gridpg)
            DATASET="imagenet"
            ;;
        epg)
            DATASET="voc"
            ;;
    esac
fi

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
    OUTPUT_DIR="${OUTPUTS_DIR}/htp/evaluation/${EVAL_TYPE}/${DATASET}"
fi

ensure_dir "${OUTPUT_DIR}"

PYTHON_DIR="$(dirname "${BASH_SOURCE[0]}")/python"

case ${EVAL_TYPE} in
    gridpg)
        CONFIDENT_DIR="${OUTPUT_DIR}/confident_images"
        GRID_DIR="${OUTPUT_DIR}/grid_pg_images_${GRID_SIZE}x${GRID_SIZE}"
        
        ensure_dir "${CONFIDENT_DIR}"
        ensure_dir "${GRID_DIR}"
        
        log_info "Step 1: Getting confident images..."
        python3 "${PYTHON_DIR}/get_confident_images.py" \
            --model-path "${MODEL_PATH}" \
            --data-root "${DATA_ROOT}" \
            --output-dir "${CONFIDENT_DIR}" \
            --confidence "${CONFIDENCE}"
        
        log_info "Step 2: Creating GridPG images..."
        python3 "${PYTHON_DIR}/create_grid_dataset.py" \
            --base-dir "${CONFIDENT_DIR}" \
            --output-dir "${GRID_DIR}" \
            --grid-size "${GRID_SIZE}"
        
        log_info "Step 3: Evaluating GridPG metrics..."
        python3 "${PYTHON_DIR}/eval_gridpg.py" \
            --model-path "${MODEL_PATH}" \
            --data-root "${GRID_DIR}" \
            --output-dir "${OUTPUT_DIR}/results" \
            --map-size "${GRID_SIZE}"
        ;;
    
    epg)
        log_info "Evaluating EPG metrics on ${DATASET}..."
        
        python3 "${PYTHON_DIR}/eval_epg.py" \
            --model-path "${MODEL_PATH}" \
            --data-root "${DATA_ROOT}" \
            --output-dir "${OUTPUT_DIR}" \
            --dataset "${DATASET}"
        ;;
    
    *)
        log_error "Unknown evaluation type: ${EVAL_TYPE}"
        exit 1
        ;;
esac

log_success "Evaluation completed"
