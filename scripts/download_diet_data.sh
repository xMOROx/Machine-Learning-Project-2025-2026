#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Download DiET datasets: Colorized MNIST, Chest X-ray, CelebA"
    echo ""
    echo "Options:"
    echo "  --mnist       Download Colorized MNIST dataset"
    echo "  --xray        Download Chest X-ray dataset (requires Kaggle API)"
    echo "  --celeba      Download CelebA dataset (requires Kaggle API)"
    echo "  --all         Download all datasets"
    echo "  --data-dir    Custom data directory (default: ${DATA_DIR})"
    echo "  -h, --help    Show this help message"
}

DOWNLOAD_MNIST=false
DOWNLOAD_XRAY=false
DOWNLOAD_CELEBA=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --mnist)
            DOWNLOAD_MNIST=true
            shift
            ;;
        --xray)
            DOWNLOAD_XRAY=true
            shift
            ;;
        --celeba)
            DOWNLOAD_CELEBA=true
            shift
            ;;
        --all)
            DOWNLOAD_MNIST=true
            DOWNLOAD_XRAY=true
            DOWNLOAD_CELEBA=true
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

if ! $DOWNLOAD_MNIST && ! $DOWNLOAD_XRAY && ! $DOWNLOAD_CELEBA; then
    log_warning "No dataset specified. Use --help for usage information."
    exit 0
fi

ensure_dir "${DATA_DIR}"
ensure_dir "${DATA_DIR}/diet"

download_colorized_mnist() {
    log_info "Downloading Colorized MNIST..."
    
    MNIST_DIR="${DATA_DIR}/diet/colorized-MNIST"
    MNIST_REPO="https://github.com/jayaneetha/colorized-MNIST.git"
    
    if [ -d "${MNIST_DIR}" ]; then
        log_warning "Colorized MNIST already exists at ${MNIST_DIR}"
        return 0
    fi
    
    git clone "${MNIST_REPO}" "${MNIST_DIR}"
    
    HARD_MNIST_DIR="${DATA_DIR}/diet/hard_mnist"
    ensure_dir "${HARD_MNIST_DIR}/training"
    ensure_dir "${HARD_MNIST_DIR}/testing"
    
    for i in {0..9}; do
        ensure_dir "${HARD_MNIST_DIR}/training/$i"
        ensure_dir "${HARD_MNIST_DIR}/testing/$i"
    done
    
    log_success "Colorized MNIST downloaded successfully"
}

download_chest_xray() {
    log_info "Downloading Chest X-ray dataset..."
    
    XRAY_DIR="${DATA_DIR}/diet/chest-xray"
    
    if [ -d "${XRAY_DIR}" ]; then
        log_warning "Chest X-ray dataset already exists at ${XRAY_DIR}"
        return 0
    fi
    
    if ! check_command kaggle; then
        log_error "Kaggle CLI is required. Install with: pip install kaggle"
        log_info "Also configure your Kaggle API credentials in ~/.kaggle/kaggle.json"
        return 1
    fi
    
    ensure_dir "${XRAY_DIR}"
    kaggle datasets download -d paulti/chest-xray-images -p "${XRAY_DIR}" --unzip
    
    log_success "Chest X-ray dataset downloaded successfully"
}

download_celeba() {
    log_info "Downloading CelebA dataset..."
    
    CELEBA_DIR="${DATA_DIR}/diet/celeba"
    
    if [ -d "${CELEBA_DIR}" ]; then
        log_warning "CelebA dataset already exists at ${CELEBA_DIR}"
        return 0
    fi
    
    if ! check_command kaggle; then
        log_error "Kaggle CLI is required. Install with: pip install kaggle"
        log_info "Also configure your Kaggle API credentials in ~/.kaggle/kaggle.json"
        return 1
    fi
    
    ensure_dir "${CELEBA_DIR}"
    kaggle datasets download -d jessicali9530/celeba-dataset -p "${CELEBA_DIR}" --unzip
    
    log_success "CelebA dataset downloaded successfully"
}

log_info "Starting DiET data download..."
log_info "Data directory: ${DATA_DIR}"

if $DOWNLOAD_MNIST; then
    download_colorized_mnist
fi

if $DOWNLOAD_XRAY; then
    download_chest_xray
fi

if $DOWNLOAD_CELEBA; then
    download_celeba
fi

log_success "DiET data download completed"
