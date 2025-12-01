#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Run How-to-Probe training pipeline"
    echo ""
    echo "Options:"
    echo "  --method        SSL method: mocov2, byol, dino (default: dino)"
    echo "  --backbone      Backbone: resnet50, bcosresnet50 (default: resnet50)"
    echo "  --probe-types   Probe types (comma-separated, default: linear,bcos-3)"
    echo "  --datasets      Datasets for probing (comma-separated, default: imagenet)"
    echo "  --loss          Loss function: bce, ce (default: bce)"
    echo "  --gpus          Number of GPUs (default: 4)"
    echo "  --skip-pretrain Skip pretraining"
    echo "  --skip-probing  Skip probing"
    echo "  --skip-eval     Skip evaluation"
    echo "  -h, --help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --method dino --backbone bcosresnet50"
    echo "  $0 --method mocov2 --probe-types linear,bcos-1,bcos-2,bcos-3"
}

bash "${SCRIPT_DIR}/htp_scripts/run_pipeline.sh" "$@"
