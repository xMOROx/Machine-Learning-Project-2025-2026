#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Run all training pipelines"
    echo ""
    echo "Options:"
    echo "  --diet          Run DiET training only"
    echo "  --htp           Run How-to-Probe training only"
    echo "  --all           Run all training pipelines (default)"
    echo "  -h, --help      Show this help message"
    echo ""
    echo "For specific options, use train_diet.sh or train_htp.sh directly"
}

RUN_DIET=false
RUN_HTP=false
RUN_ALL=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --diet)
            RUN_DIET=true
            RUN_ALL=false
            shift
            ;;
        --htp)
            RUN_HTP=true
            RUN_ALL=false
            shift
            ;;
        --all)
            RUN_ALL=true
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

if $RUN_ALL; then
    RUN_DIET=true
    RUN_HTP=true
fi

log_info "============================================"
log_info "Training Orchestration"
log_info "============================================"

if $RUN_DIET; then
    log_info ""
    log_info ">>> Running DiET training pipeline..."
    bash "${SCRIPT_DIR}/train_diet.sh" --dataset mnist
fi

if $RUN_HTP; then
    log_info ""
    log_info ">>> Running How-to-Probe training pipeline..."
    bash "${SCRIPT_DIR}/train_htp.sh" --method dino --backbone resnet50
fi

log_info ""
log_info "============================================"
log_success "All training completed"
log_info "============================================"
