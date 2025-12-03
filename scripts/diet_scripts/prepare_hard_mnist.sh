#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/config.sh"

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Prepare Hard MNIST dataset from Colorized MNIST"
    echo ""
    echo "Options:"
    echo "  --data-dir    Custom data directory (default: ${DATA_DIR}/diet)"
    echo "  -h, --help    Show this help message"
}

while [[ $# -gt 0 ]]; do
    case $1 in
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

DIET_DATA_DIR="${DATA_DIR}/diet"
COLORIZED_MNIST_DIR="${DIET_DATA_DIR}/colorized-MNIST"
HARD_MNIST_DIR="${DIET_DATA_DIR}/hard_mnist"

if [ ! -d "${COLORIZED_MNIST_DIR}" ]; then
    log_error "Colorized MNIST not found at ${COLORIZED_MNIST_DIR}"
    log_info "Run download_diet_data.sh --mnist first"
    exit 1
fi

log_info "Preparing Hard MNIST dataset..."

cd "${PROJECT_ROOT}"

python3 "${SCRIPT_DIR}/diet_scripts/python/prepare_hard_mnist.py" \
    --input-dir "${COLORIZED_MNIST_DIR}" \
    --output-dir "${HARD_MNIST_DIR}"

log_success "Hard MNIST dataset prepared at ${HARD_MNIST_DIR}"
