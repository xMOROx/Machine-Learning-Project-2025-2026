#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Download all datasets for the project"
    echo ""
    echo "Options:"
    echo "  --diet        Download DiET datasets only"
    echo "  --htp         Download How-to-Probe datasets only"
    echo "  --project     Download project-specific datasets only"
    echo "  --all         Download all datasets (default)"
    echo "  --data-dir    Custom data directory (default: ${DATA_DIR})"
    echo "  -h, --help    Show this help message"
}

DOWNLOAD_DIET=false
DOWNLOAD_HTP=false
DOWNLOAD_PROJECT=false
DOWNLOAD_ALL=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --diet)
            DOWNLOAD_DIET=true
            DOWNLOAD_ALL=false
            shift
            ;;
        --htp)
            DOWNLOAD_HTP=true
            DOWNLOAD_ALL=false
            shift
            ;;
        --project)
            DOWNLOAD_PROJECT=true
            DOWNLOAD_ALL=false
            shift
            ;;
        --all)
            DOWNLOAD_ALL=true
            shift
            ;;
        --data-dir)
            DATA_DIR="$2"
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

if $DOWNLOAD_ALL; then
    DOWNLOAD_DIET=true
    DOWNLOAD_HTP=true
    DOWNLOAD_PROJECT=true
fi

log_info "============================================"
log_info "Starting data download orchestration"
log_info "Data directory: ${DATA_DIR}"
log_info "============================================"

if $DOWNLOAD_DIET; then
    log_info ""
    log_info ">>> Downloading DiET datasets..."
    bash "${SCRIPT_DIR}/download_diet_data.sh" --all --data-dir "${DATA_DIR}"
fi

if $DOWNLOAD_HTP; then
    log_info ""
    log_info ">>> Downloading How-to-Probe datasets..."
    bash "${SCRIPT_DIR}/download_htp_data.sh" --coco --voc --data-dir "${DATA_DIR}"
fi

if $DOWNLOAD_PROJECT; then
    log_info ""
    log_info ">>> Downloading project datasets..."
    bash "${SCRIPT_DIR}/download_project_data.sh" --all --data-dir "${DATA_DIR}"
fi

log_info ""
log_info "============================================"
log_success "All data downloads completed"
log_info "============================================"
