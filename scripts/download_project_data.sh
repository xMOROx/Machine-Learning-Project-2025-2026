#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Download project-specific datasets: CIFAR-10, GLUE SST-2, Adult Census"
    echo ""
    echo "Options:"
    echo "  --cifar10     Download CIFAR-10 dataset"
    echo "  --glue        Download GLUE SST-2 dataset"
    echo "  --adult       Download Adult Census dataset"
    echo "  --all         Download all datasets"
    echo "  --data-dir    Custom data directory (default: ${DATA_DIR})"
    echo "  -h, --help    Show this help message"
}

DOWNLOAD_CIFAR=false
DOWNLOAD_GLUE=false
DOWNLOAD_ADULT=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --cifar10)
            DOWNLOAD_CIFAR=true
            shift
            ;;
        --glue)
            DOWNLOAD_GLUE=true
            shift
            ;;
        --adult)
            DOWNLOAD_ADULT=true
            shift
            ;;
        --all)
            DOWNLOAD_CIFAR=true
            DOWNLOAD_GLUE=true
            DOWNLOAD_ADULT=true
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

if ! $DOWNLOAD_CIFAR && ! $DOWNLOAD_GLUE && ! $DOWNLOAD_ADULT; then
    log_warning "No dataset specified. Use --help for usage information."
    exit 0
fi

ensure_dir "${DATA_DIR}"

download_cifar10() {
    log_info "Downloading CIFAR-10..."
    
    CIFAR_DIR="${DATA_DIR}/cifar10"
    
    python3 << EOF
import torchvision
import os

path = "${CIFAR_DIR}"
os.makedirs(path, exist_ok=True)

print(f"Downloading CIFAR-10 to {path}...")
torchvision.datasets.CIFAR10(root=path, train=True, download=True)
torchvision.datasets.CIFAR10(root=path, train=False, download=True)
print("CIFAR-10 download complete.")
EOF
    
    log_success "CIFAR-10 downloaded successfully"
}

download_glue_sst2() {
    log_info "Downloading GLUE SST-2..."
    
    GLUE_DIR="${DATA_DIR}/glue_sst2"
    
    python3 << EOF
from datasets import load_dataset
import os

path = "${GLUE_DIR}"
os.makedirs(path, exist_ok=True)

print(f"Downloading GLUE SST-2 to {path}...")
load_dataset('glue', 'sst2', cache_dir=path)
print("GLUE SST-2 download complete.")
EOF
    
    log_success "GLUE SST-2 downloaded successfully"
}

download_adult() {
    log_info "Downloading Adult Census dataset..."
    
    ADULT_DIR="${DATA_DIR}/adult"
    ensure_dir "${ADULT_DIR}"
    
    BASE_URL="https://archive.ics.uci.edu/ml/machine-learning-databases/adult"
    FILES=("adult.data" "adult.test" "adult.names")
    
    for file in "${FILES[@]}"; do
        if [ ! -f "${ADULT_DIR}/${file}" ]; then
            log_info "Downloading ${file}..."
            curl -sL -o "${ADULT_DIR}/${file}" "${BASE_URL}/${file}" || \
                wget -q -O "${ADULT_DIR}/${file}" "${BASE_URL}/${file}"
        else
            log_warning "${file} already exists, skipping."
        fi
    done
    
    log_success "Adult Census downloaded successfully"
}

log_info "Starting project data download..."
log_info "Data directory: ${DATA_DIR}"

if $DOWNLOAD_CIFAR; then
    download_cifar10
fi

if $DOWNLOAD_GLUE; then
    download_glue_sst2
fi

if $DOWNLOAD_ADULT; then
    download_adult
fi

log_success "Project data download completed"
