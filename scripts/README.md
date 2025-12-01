# Scripts

This directory contains scripts for data downloading, model training, and evaluation.

## Directory Structure

```
scripts/
├── config.sh                 # Shared configuration and utilities
├── download_all_data.sh      # Orchestrates all data downloads
├── download_diet_data.sh     # Download DiET datasets
├── download_htp_data.sh      # Download How-to-Probe datasets
├── download_project_data.sh  # Download project-specific datasets
├── train_all.sh              # Run all training pipelines
├── train_diet.sh             # Run DiET training pipeline
├── train_htp.sh              # Run How-to-Probe training pipeline
├── diet_scripts/             # DiET-specific scripts
│   ├── run_pipeline.sh       # Complete DiET pipeline
│   ├── prepare_hard_mnist.sh # Prepare Hard MNIST dataset
│   ├── train_baseline.sh     # Train baseline models
│   ├── distillation.sh       # Run DiET distillation
│   ├── inference.sh          # Run inference on distilled model
│   └── evaluate.sh           # Evaluate with metrics
└── htp_scripts/              # How-to-Probe specific scripts
    ├── run_pipeline.sh       # Complete HTP pipeline
    ├── pretrain.sh           # SSL pretraining (MoCov2, BYOL, DINO)
    ├── probe.sh              # Train linear/MLP probes
    └── evaluate.sh           # Evaluate attributions (GridPG, EPG)
```

## Quick Start

### Download Data

```bash
# Download all datasets
./scripts/download_all_data.sh --all

# Download specific datasets
./scripts/download_diet_data.sh --mnist
./scripts/download_htp_data.sh --coco --voc
./scripts/download_project_data.sh --cifar10 --adult
```

### Train Models

```bash
# Run complete DiET pipeline
./scripts/train_diet.sh --dataset mnist --ups 4

# Run complete How-to-Probe pipeline
./scripts/train_htp.sh --method dino --backbone resnet50

# Run all training
./scripts/train_all.sh
```

## Data Download Scripts

### download_diet_data.sh

Downloads datasets for DiET:
- `--mnist`: Colorized MNIST (clones from GitHub)
- `--xray`: Chest X-ray (requires Kaggle API)
- `--celeba`: CelebA (requires Kaggle API)

### download_htp_data.sh

Downloads datasets for How-to-Probe:
- `--imagenet`: ImageNet (manual download required)
- `--coco`: COCO 2014
- `--voc`: Pascal VOC 2007/2012

### download_project_data.sh

Downloads project-specific datasets:
- `--cifar10`: CIFAR-10
- `--glue`: GLUE SST-2
- `--adult`: Adult Census

## Training Scripts

### DiET Pipeline (train_diet.sh)

```bash
./scripts/train_diet.sh [OPTIONS]

Options:
  --dataset       Dataset: mnist, xray, celeba (default: mnist)
  --ups           Upsampling factor (default: 4)
  --lr            Learning rate for distillation (default: 2000)
  --epochs        Epochs for baseline training (default: 10)
  --skip-baseline Skip baseline training
  --skip-distill  Skip distillation
  --skip-eval     Skip evaluation
  --gpu           GPU device ID (default: 0)
```

### How-to-Probe Pipeline (train_htp.sh)

```bash
./scripts/train_htp.sh [OPTIONS]

Options:
  --method        SSL method: mocov2, byol, dino (default: dino)
  --backbone      Backbone: resnet50, bcosresnet50 (default: resnet50)
  --probe-types   Probe types (comma-separated, default: linear,bcos-3)
  --datasets      Datasets for probing (comma-separated, default: imagenet)
  --loss          Loss function: bce, ce (default: bce)
  --gpus          Number of GPUs (default: 4)
  --skip-pretrain Skip pretraining
  --skip-probing  Skip probing
  --skip-eval     Skip evaluation
```

## Individual Scripts

### DiET Scripts

```bash
# Prepare Hard MNIST from Colorized MNIST
./scripts/diet_scripts/prepare_hard_mnist.sh

# Train baseline model
./scripts/diet_scripts/train_baseline.sh --dataset mnist --epochs 10

# Run distillation
./scripts/diet_scripts/distillation.sh --dataset mnist --ups 4 --lr 2000

# Run inference
./scripts/diet_scripts/inference.sh --dataset mnist --ups 4

# Evaluate
./scripts/diet_scripts/evaluate.sh --dataset mnist --eval-type all
```

### How-to-Probe Scripts

```bash
# Pretrain SSL model
./scripts/htp_scripts/pretrain.sh --method dino --backbone resnet50

# Train probe
./scripts/htp_scripts/probe.sh --dataset imagenet --probe-type linear --loss bce

# Evaluate attributions
./scripts/htp_scripts/evaluate.sh --eval-type gridpg --dataset imagenet
```

## Configuration

Edit `config.sh` to customize paths:

```bash
DATA_DIR="${PROJECT_ROOT}/data"
OUTPUTS_DIR="${PROJECT_ROOT}/outputs"
MODELS_DIR="${PROJECT_ROOT}/outputs/models"
```

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- For Kaggle datasets: `pip install kaggle` and configure API credentials
- For How-to-Probe: mmpretrain dependencies

## Output Structure

```
outputs/
├── diet/
│   ├── trained_models/
│   │   ├── hard_mnist_rn34.pth
│   │   └── ...
│   └── distillation/
│       ├── mnist_ups4_outdir/
│       └── ...
└── htp/
    ├── pretraining/
    │   ├── dino_resnet50/
    │   └── ...
    ├── probing/
    │   ├── imagenet/
    │   └── ...
    └── evaluation/
        ├── gridpg/
        └── epg/
```
