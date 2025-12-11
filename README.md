# XAI: Tuning Models for Faithful and Discriminative Attributions

![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 1. Project Overview

Standard post-hoc explanation methods (e.g., GradCAM, Integrated Gradients) are a popular tool for interpreting deep neural networks. However, they often suffer from a critical flaw: their attributions can be **unfaithful** to the model's underlying reasoning. 'An explanation might highlight features that are inconsistent with the model's actual behavior or non-discriminative for the task'.

This project implements and analyzes two recent, powerful techniques designed to bridge this gap and produce more faithful explanations:

1. **Distractor Erasure Tuning (DiET)**: A method that fine-tunes a black-box model to be robust to the erasure of "distractor" features (i.e., non-discriminative parts of the input). This process yields a new model that is both faithful to the original and produces highly discriminative attributions[cite: 3306, 3408].
2. **"How to Probe" Framework**: An analysis framework demonstrating that the design of a model's classification head (the "probe")—even if it's <10% of the parameters—has a crucial impact on the quality of post-hoc explanations[cite: 18]. Simply changing the probe's loss function (e.g., from Cross-Entropy to **BCE**) or architecture (e.g., **Linear** to **MLP**) can significantly improve attribution localization.

This repository serves as a unified benchmark to reproduce, compare, and analyze these state-of-the-art methods for improving explanation faithfulness.

## 2. Core Papers

This project is a reproduction and comparative analysis of the following papers:

* **(DiET)**: Bhalla, U., et al. (2023). **"Discriminative Feature Attributions: Bridging Post Hoc Explainability and Inherent Interpretability."** *NeurIPS 2023.*
  * [Paper PDF](<https://proceedings.neurips.cc/paper_files/paper/2023/file/5529f5f08d6d366b5c3e6f988900693b-Paper-Conference.pdf>)
  * [Official Repo](<https://github.com/AI4LIFE-GROUP/DiET>)
* **(How to Probe)**: Gairola, S., et al. (2025). **"HOW TO PROBE: SIMPLE YET EFFECTIVE TECHNIQUES FOR IMPROVING POST-HOC EXPLANATIONS."** *ICLR 2025.*
  * [Paper PDF](<https://openreview.net/pdf?id=66J13t3i3E>)
  * [Official Repo](<https://github.com/sidgairo18/how-to-probe>)

## 3. Setup & Installation

This project uses Python `venv` for environment management and `pip` for package installation.

### Step 1: Clone This Repository

```bash
git clone --recursive https://github.com/xMOROx/Machine-Learning-Project-2025-2026.git
cd Machine-Learning-Project-2025-2026
```

### Step 2: Create and Activate a Virtual Environment

```bash
# Create the virtual environment
python -m venv .venv

# Activate the environment
# On macOS / Linux:
source .venv/bin/activate
# On Windows (PowerShell):
# .\.venv\Scripts\Activate.ps1
```

### Step 3: Install PyTorch

Follow the instructions on the [PyTorch website](https://pytorch.org/get-started/locally/) to install the appropriate version for your system. For example, for CUDA 11.8, you can run:

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
```

### Step 4: Clone Required Project Repositories

```bash
git submodule update --init --recursive
```

### Step 5: Install Project Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Verify Installation

```bash
python scripts/verify.py
```

## 4. Data Download

```bash
bash scripts/download_project_data.sh --all
```

## 5. XAI Experiments

The project includes comprehensive XAI (Explainable AI) experiments with multiple models and explanation methods optimized for RTX 3060 (12GB VRAM).

### 5.1 Available Experiments

| Experiment | Description | XAI Method |
|------------|-------------|------------|
| **CIFAR-10** | CNN classification with visual explanations | GradCAM |
| **GLUE SST-2** | BERT sentiment analysis with text explanations | Integrated Gradients |
| **Model Comparison** | Compare CNN, RF, LightGBM, SVM, Logistic Regression | - |

### 5.2 Running XAI Experiments

**Run all experiments:**
```bash
./scripts/run.sh xai --xai-exp all
```

**Run specific experiments:**
```bash
# CIFAR-10 with GradCAM
./scripts/run.sh xai --xai-exp cifar10 --epochs 10

# GLUE SST-2 with BERT
./scripts/run.sh xai --xai-exp glue --epochs 3

# Model comparison
./scripts/run.sh xai --xai-exp compare
```

**For RTX 3060 or low VRAM GPUs:**
```bash
./scripts/run.sh xai --xai-exp all --low-vram
```

**Direct Python usage:**
```bash
cd scripts/xai_experiments
python run_xai_experiments.py --all --low-vram
```

### 5.3 XAI Outputs

Results are saved to `outputs/xai_experiments/`:

```
outputs/xai_experiments/
├── cifar10/
│   ├── resnet_cifar10.pth          # Trained model
│   ├── gradcam_batch.png           # Batch visualization
│   ├── gradcam_visualizations/     # Individual GradCAM images
│   └── experiment_results.json     # Metrics and results
├── glue_sst2/
│   ├── bert_sst2/                  # Fine-tuned BERT model
│   ├── ig_visualizations/          # Token attribution images
│   └── experiment_results.json     # Metrics and results
└── model_comparison/
    └── comparison_results.json     # Model comparison metrics
```

### 5.4 XAI Methods Implemented

#### GradCAM (Gradient-weighted Class Activation Mapping)
- Visualizes which regions of an image are important for CNN predictions
- Produces heatmaps overlaid on original images
- Suitable for any CNN architecture

#### Integrated Gradients
- Attributes prediction to input features along a path from baseline
- For text: highlights important tokens for classification
- Satisfies axioms of sensitivity and implementation invariance

### 5.5 Model Architectures

| Model | Type | Description |
|-------|------|-------------|
| SimpleCNN | Vision | 4-layer CNN optimized for CIFAR-10 |
| ResNetCIFAR | Vision | ResNet-18 adapted for 32x32 images |
| BERT | Language | bert-base-uncased for sentiment analysis |
| Random Forest | Traditional ML | Ensemble of decision trees |
| LightGBM | Traditional ML | Gradient boosting framework |
| SVM | Traditional ML | Support Vector Machine with RBF kernel |
| Logistic Regression | Traditional ML | Linear classifier |

### 5.6 Hardware Optimization

The orchestration script automatically detects GPU memory and adjusts settings:

| GPU Memory | Batch Size | Epochs | Workers |
|------------|------------|--------|---------|
| ≥ 8GB | 64 | 10 | 4 |
| < 8GB | 16 | 5 | 2 |
| CPU only | 16 | 2 | 0 |

Use `--low-vram` flag to force low memory configuration.