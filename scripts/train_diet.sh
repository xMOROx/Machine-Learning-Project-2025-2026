#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Run DiET training pipeline"
    echo ""
    echo "Options:"
    echo "  --dataset       Dataset: mnist, xray, celeba (default: mnist)"
    echo "  --ups           Upsampling factor (default: 4)"
    echo "  --lr            Learning rate for distillation (default: 2000)"
    echo "  --epochs        Epochs for baseline training (default: 10)"
    echo "  --skip-baseline Skip baseline training"
    echo "  --skip-distill  Skip distillation"
    echo "  --skip-eval     Skip evaluation"
    echo "  --gpu           GPU device ID (default: 0)"
    echo "  -h, --help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --dataset mnist --ups 4"
    echo "  $0 --dataset xray --epochs 20 --gpu 1"
}

bash "${SCRIPT_DIR}/diet_scripts/run_pipeline.sh" "$@"
