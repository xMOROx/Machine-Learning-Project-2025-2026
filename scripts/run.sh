#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

print_usage() {
  echo "Usage: $0 <command> [OPTIONS]"
  echo ""
  echo "Unified orchestration script for Machine Learning Project 2025-2026"
  echo ""
  echo "Commands:"
  echo "  download          Download datasets"
  echo "  diet              Run DiET pipeline"
  echo "  htp               Run How-to-Probe pipeline"
  echo "  all               Run everything (download + diet + htp)"
  echo ""
  echo "Download options:"
  echo "  --cifar10         Download CIFAR-10 dataset"
  echo "  --glue            Download GLUE SST-2 dataset"
  echo "  --mnist           Download Colorized MNIST dataset"
  echo "  --adult           Download Adult Census dataset"
  echo "  --all             Download all datasets"
  echo ""
  echo "DiET options:"
  echo "  --dataset         Dataset: mnist, xray, celeba (default: mnist)"
  echo "  --epochs          Training epochs (default: 10)"
  echo "  --ups             Upsampling factor (default: 4)"
  echo "  --lr              Learning rate for distillation (default: 2000)"
  echo "  --skip-prepare    Skip data preparation"
  echo "  --skip-train      Skip baseline training"
  echo "  --skip-distill    Skip distillation"
  echo "  --skip-eval       Skip evaluation"
  echo ""
  echo "HTP options:"
  echo "  --method          SSL method: mocov2, byol, dino (default: dino)"
  echo "  --backbone        Backbone: resnet50, bcosresnet50 (default: resnet50)"
  echo "  --probe-type      Probe type: linear, bcos-1, std-2, etc. (default: linear)"
  echo "  --skip-pretrain   Skip pretraining"
  echo "  --skip-probe      Skip probing"
  echo "  --skip-eval       Skip evaluation"
  echo ""
  echo "Common options:"
  echo "  --gpu             GPU device ID (default: 0)"
  echo "  --gpus            Number of GPUs for distributed training (default: 4)"
  echo "  --low-vram        Use low VRAM settings for small laptops"
  echo "  --dry-run         Print commands without executing"
  echo "  -h, --help        Show this help message"
  echo ""
  echo "Examples:"
  echo "  $0 download --all"
  echo "  $0 download --cifar10 --mnist"
  echo "  $0 diet --dataset mnist --epochs 10"
  echo "  $0 diet --dataset mnist --low-vram   # For small GPUs"
  echo "  $0 htp --method dino --backbone resnet50"
  echo "  $0 all --gpu 0"
}

COMMAND=""
DOWNLOAD_CIFAR=false
DOWNLOAD_GLUE=false
DOWNLOAD_MNIST=false
DOWNLOAD_ADULT=false
DATASET="mnist"
EPOCHS=10
UPS=4
LR=2000
METHOD="dino"
BACKBONE="resnet50"
PROBE_TYPE="linear"
GPU=0
GPUS=1
DRY_RUN=false
LOW_VRAM=false
SKIP_PREPARE=false
SKIP_TRAIN=false
SKIP_DISTILL=false
SKIP_PRETRAIN=false
SKIP_PROBE=false
SKIP_EVAL=false

if [ $# -eq 0 ]; then
  print_usage
  exit 0
fi

COMMAND="$1"
shift

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
  --mnist)
    DOWNLOAD_MNIST=true
    shift
    ;;
  --adult)
    DOWNLOAD_ADULT=true
    shift
    ;;
  --all)
    DOWNLOAD_CIFAR=true
    DOWNLOAD_GLUE=true
    DOWNLOAD_MNIST=true
    DOWNLOAD_ADULT=true
    shift
    ;;
  --dataset)
    DATASET="$2"
    shift 2
    ;;
  --epochs)
    EPOCHS="$2"
    shift 2
    ;;
  --ups)
    UPS="$2"
    shift 2
    ;;
  --lr)
    LR="$2"
    shift 2
    ;;
  --method)
    METHOD="$2"
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
  --gpu)
    GPU="$2"
    shift 2
    ;;
  --gpus)
    GPUS="$2"
    shift 2
    ;;
  --dry-run)
    DRY_RUN=true
    shift
    ;;
  --low-vram)
    LOW_VRAM=true
    export LOW_VRAM=true
    shift
    ;;
  --skip-prepare)
    SKIP_PREPARE=true
    shift
    ;;
  --skip-train)
    SKIP_TRAIN=true
    shift
    ;;
  --skip-distill)
    SKIP_DISTILL=true
    shift
    ;;
  --skip-pretrain)
    SKIP_PRETRAIN=true
    shift
    ;;
  --skip-probe)
    SKIP_PROBE=true
    shift
    ;;
  --skip-eval)
    SKIP_EVAL=true
    shift
    ;;
  -h | --help)
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

run_cmd() {
  if [ "${DRY_RUN}" = true ]; then
    echo "[DRY-RUN] $*"
  else
    "$@"
  fi
}

cmd_download() {
  log_info "=== Downloading Datasets ==="

  if ! $DOWNLOAD_CIFAR && ! $DOWNLOAD_GLUE && ! $DOWNLOAD_MNIST && ! $DOWNLOAD_ADULT; then
    log_error "Specify what to download: --cifar10, --glue, --mnist, --adult, or --all"
    exit 1
  fi

  DOWNLOAD_ARGS=""
  if [ "${DOWNLOAD_CIFAR}" = true ]; then
    DOWNLOAD_ARGS="${DOWNLOAD_ARGS} --cifar10"
  fi
  if [ "${DOWNLOAD_GLUE}" = true ]; then
    DOWNLOAD_ARGS="${DOWNLOAD_ARGS} --glue"
  fi
  if [ "${DOWNLOAD_MNIST}" = true ]; then
    DOWNLOAD_ARGS="${DOWNLOAD_ARGS} --mnist"
  fi
  if [ "${DOWNLOAD_ADULT}" = true ]; then
    DOWNLOAD_ARGS="${DOWNLOAD_ARGS} --adult"
  fi

  run_cmd "${SCRIPT_DIR}/download_project_data.sh" ${DOWNLOAD_ARGS}

  log_success "Download completed"
}

# ...existing code...

cmd_diet() {
  log_info "=== Running DiET Pipeline ==="
  log_info "Dataset: ${DATASET}"
  log_info "Epochs: ${EPOCHS}"
  log_info "Upsampling: ${UPS}"
  log_info "Learning rate: ${LR}"
  log_info "GPU: ${GPU}"
  if [ "${LOW_VRAM}" = true ]; then
    log_info "Low VRAM mode: enabled"
    LOW_VRAM_FLAG="--low-vram"
  else
    LOW_VRAM_FLAG=""
  fi

  if [ "${SKIP_PREPARE}" = false ] && [ "${DATASET}" = "mnist" ]; then
    log_info "Step 1: Preparing Hard MNIST..."
    run_cmd "${SCRIPT_DIR}/diet_scripts/prepare_hard_mnist.sh"
  fi

  if [ "${SKIP_TRAIN}" = false ]; then
    log_info "Step 2: Training baseline model..."
    run_cmd "${SCRIPT_DIR}/diet_scripts/train_baseline.sh" \
      --dataset "${DATASET}" \
      --epochs "${EPOCHS}" \
      --gpu "${GPU}" \
      ${LOW_VRAM_FLAG}
  fi

  if [ "${SKIP_DISTILL}" = false ]; then
    log_info "Step 3: Running distillation..."
    run_cmd "${SCRIPT_DIR}/diet_scripts/distillation.sh" \
      --dataset "${DATASET}" \
      --ups "${UPS}" \
      --lr "${LR}" \
      --gpu "${GPU}" \
      ${LOW_VRAM_FLAG}

    log_info "Step 4: Running inference..."
    run_cmd "${SCRIPT_DIR}/diet_scripts/inference.sh" \
      --dataset "${DATASET}" \
      --ups "${UPS}" \
      --gpu "${GPU}" \
      ${LOW_VRAM_FLAG}
  fi

  if [ "${SKIP_EVAL}" = false ]; then
    log_info "Step 5: Running evaluation..."
    run_cmd "${SCRIPT_DIR}/diet_scripts/evaluate.sh" \
      --dataset "${DATASET}" \
      --ups "${UPS}" \
      --eval-type all \
      --gpu "${GPU}" \
      ${LOW_VRAM_FLAG}
  fi

  log_success "DiET pipeline completed"
}

cmd_htp() {
  log_info "=== Running How-to-Probe Pipeline ==="
  log_info "Method: ${METHOD}"
  log_info "Backbone: ${BACKBONE}"
  log_info "Probe type: ${PROBE_TYPE}"
  log_info "GPUs: ${GPUS}"

  if [ "${SKIP_PRETRAIN}" = false ]; then
    log_info "Step 1: Pretraining..."
    run_cmd "${SCRIPT_DIR}/htp_scripts/pretrain.sh" \
      --method "${METHOD}" \
      --backbone "${BACKBONE}" \
      --gpus "${GPUS}"
  fi

  if [ "${SKIP_PROBE}" = false ]; then
    log_info "Step 2: Training probes..."
    run_cmd "${SCRIPT_DIR}/htp_scripts/probe.sh" \
      --backbone "${BACKBONE}" \
      --probe-type "${PROBE_TYPE}" \
      --gpus "${GPUS}"
  fi

  if [ "${SKIP_EVAL}" = false ]; then
    log_info "Step 3: Evaluating GridPG..."
    run_cmd "${SCRIPT_DIR}/htp_scripts/evaluate.sh" \
      --eval-type gridpg

    log_info "Step 4: Evaluating EPG..."
    run_cmd "${SCRIPT_DIR}/htp_scripts/evaluate.sh" \
      --eval-type epg
  fi

  log_success "How-to-Probe pipeline completed"
}

cmd_all() {
  log_info "=== Running Full Pipeline ==="

  DOWNLOAD_DIET=true
  DOWNLOAD_HTP=true
  DOWNLOAD_PROJECT=true
  cmd_download

  cmd_diet
  cmd_htp

  log_success "Full pipeline completed"
}

case ${COMMAND} in
download)
  cmd_download
  ;;
diet)
  cmd_diet
  ;;
htp)
  cmd_htp
  ;;
all)
  cmd_all
  ;;
-h | --help | help)
  print_usage
  ;;
*)
  log_error "Unknown command: ${COMMAND}"
  print_usage
  exit 1
  ;;
esac
