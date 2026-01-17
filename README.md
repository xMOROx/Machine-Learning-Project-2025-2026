# XAI: DiET vs Basic Methods Comparison Framework

![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 1. Project Overview

This project provides a **comprehensive framework for comparing DiET (Discriminative Feature Attribution)** with basic XAI methods across **multiple datasets**:

### Supported Datasets

| Modality | Dataset | Description | Classes |
|----------|---------|-------------|---------|
| **Image** | CIFAR-10 | Natural images | 10 |
| **Image** | CIFAR-100 | Fine-grained images | 100 |
| **Image** | SVHN | Street View House Numbers | 10 |
| **Image** | Fashion-MNIST | Fashion products | 10 |
| **Text** | SST-2 | Sentiment analysis | 2 |
| **Text** | IMDB | Movie reviews | 2 |
| **Text** | AG News | News classification | 4 |

### Comparison Methods

- **Images**: DiET vs GradCAM
- **Text**: DiET vs Integrated Gradients

Standard post-hoc explanation methods (e.g., GradCAM, Integrated Gradients) are popular tools for interpreting deep neural networks. However, they often suffer from a critical flaw: their attributions can be **unfaithful** to the model's underlying reasoning.

**DiET (Distractor Erasure Tuning)** addresses this by fine-tuning models to be robust to the erasure of "distractor" features (non-discriminative parts of the input), yielding models that produce highly discriminative attributions.

### Key Features

- **Robust Metrics**: Pixel Perturbation, AOPC, Insertion/Deletion, Faithfulness Correlation
- **Rich Visualizations**: Bar charts, radar plots, comparison dashboards, HTML reports
- **Notebook-Friendly API**: Easy to use in Jupyter notebooks
- **Hardware Optimized**: Automatic GPU detection with low-VRAM support
- **Multi-Dataset Support**: Compare across 4 image and 3 text datasets
- **Google Colab Ready**: Pre-built notebooks for GPU-accelerated experiments

## 2. Quick Start

### Google Colab (Recommended for Fast Training)

For the fastest experience with GPU acceleration, use our pre-built Google Colab notebooks:

| Notebook | Description | Open in Colab |
|----------|-------------|---------------|
| **DiET vs GradCAM** | Image classification comparison on CIFAR-10, CIFAR-100, SVHN, Fashion-MNIST | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xMOROx/Machine-Learning-Project-2025-2026/blob/main/notebooks/DiET_vs_GradCAM_Image_Comparison.ipynb) |
| **DiET vs Integrated Gradients** | Text classification comparison on SST-2, IMDB, AG News | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xMOROx/Machine-Learning-Project-2025-2026/blob/main/notebooks/DiET_vs_IntegratedGradients_Text_Comparison.ipynb) |

**Features of Colab notebooks:**
- 🚀 Automatic GPU detection and configuration
- 📊 Comprehensive visualizations and statistical analysis
- 📄 Downloadable results (JSON, CSV, images)
- 📋 Academic-style reports suitable for presentations

### Command Line

### Command Line

```bash
# Run full DiET comparison on ALL datasets (recommended)
python scripts/xai_experiments/run_xai_experiments.py --diet

# Run only image comparison (DiET vs GradCAM) on all image datasets
python scripts/xai_experiments/run_xai_experiments.py --diet --diet-images

# Run only text comparison (DiET vs IG) on all text datasets
python scripts/xai_experiments/run_xai_experiments.py --diet --diet-text

# Low VRAM mode for smaller GPUs
python scripts/xai_experiments/run_xai_experiments.py --diet --low-vram
```

### In Jupyter Notebook

```python
from scripts.xai_experiments import XAIMethodsComparison, ComparisonConfig

# Configure the comparison with multiple datasets
config = ComparisonConfig(
    device="cuda",
    # Image datasets to compare
    image_datasets=["cifar10", "cifar100", "svhn", "fashion_mnist"],
    image_comparison_samples=100,
    # Text datasets to compare
    text_datasets=["sst2", "imdb", "ag_news"],
    text_comparison_samples=50
)

# Run comparison
comparison = XAIMethodsComparison(config)
results = comparison.run_full_comparison(run_images=True, run_text=True)

# Generate visualizations
comparison.visualize_results()

# Get results as DataFrame (one row per dataset)
df = comparison.get_results_dataframe()
print(df)
```

## 3. Reference Paper

* **(DiET)**: Bhalla, U., et al. (2023). **"Discriminative Feature Attributions: Bridging Post Hoc Explainability and Inherent Interpretability."** *NeurIPS 2023.*
  * [Paper PDF](<https://proceedings.neurips.cc/paper_files/paper/2023/file/5529f5f08d6d366b5c3e6f988900693b-Paper-Conference.pdf>)
  * [Official Repo](<https://github.com/AI4LIFE-GROUP/DiET>)

## 4. Setup & Installation

This project uses Python `uv` to manage venvs and dependencies.

### Step 1: Clone This Repository

```bash
git clone --recursive https://github.com/xMOROx/Machine-Learning-Project-2025-2026.git
cd Machine-Learning-Project-2025-2026
```

### Step 2: Create and Activate a Virtual Environment

```bash
# Create the virtual environment
uv venv
```

### Step 3: Clone Required Project Repositories

```bash
git submodule update --init --recursive
```

### Step 4: Install Project Dependencies

```bash
uv sync
```

## 5. DiET Comparison Framework

### 5.1 Experiments Overview

| Experiment | Methods Compared | Dataset | Metrics |
|------------|-----------------|---------|---------|
| **Image Comparison** | DiET vs GradCAM | CIFAR-10 | Pixel Perturbation, AOPC, Faithfulness |
| **Text Comparison** | DiET vs Integrated Gradients | SST-2 | Top-k Token Overlap, Accuracy |

### 5.2 Running Comparisons

**Full comparison (recommended):**
```bash
./scripts/run_xai.sh
```

**Image-only comparison:**
```bash
python scripts/xai_experiments/run_xai_experiments.py --diet --diet-images
```

**Text-only comparison:**
```bash
python scripts/xai_experiments/run_xai_experiments.py --diet --diet-text
```

### 5.3 Output Structure

Results are saved to `outputs/xai_experiments/diet_comparison/`:

```
outputs/xai_experiments/diet_comparison/
├── cifar10/
│   ├── baseline_resnet.pth           # Trained baseline model
│   ├── diet_resnet.pth               # DiET fine-tuned model
│   ├── diet_mask_step*.pt            # Learned DiET masks
│   ├── comparison_visualizations/    # GradCAM vs DiET heatmaps
│   └── diet_experiment_results.json  # Detailed metrics
├── sst2/
│   ├── bert_baseline/                # Fine-tuned BERT model
│   ├── diet_token_mask.pt            # Learned token masks
│   ├── comparison_visualizations/    # IG vs DiET token attributions
│   └── diet_text_results.json        # Comparison results
├── comparison_dashboard.png          # Visual summary
├── comparison_report.html            # Interactive HTML report
└── comparison_results.json           # All metrics
```

### 5.4 Metrics Computed

#### Image Metrics
- **Pixel Perturbation**: Keep/remove top-k% pixels and measure accuracy
- **AOPC (Area Over Perturbation Curve)**: Average accuracy drop over perturbations
- **Insertion/Deletion Curves**: Progressive pixel insertion/deletion
- **Faithfulness Correlation**: Correlation between attributions and model sensitivity

#### Text Metrics
- **Top-k Token Overlap**: Agreement between methods on important tokens
- **Accuracy Preservation**: Model accuracy after DiET tuning

### 5.5 Visualization Examples

The framework generates:
- **Metric comparison bar charts**
- **Radar plots for multi-metric comparison**
- **Side-by-side attribution heatmaps**
- **Perturbation curves**
- **Interactive HTML reports**

### 5.6 Hardware Optimization

The framework automatically detects GPU memory and adjusts settings:

| GPU Memory | Image Batch | Text Batch | Comparison Samples | Configuration |
|------------|-------------|------------|-------------------|---------------|
| ≥ 8GB | 64 | 16 | 100 | Standard |
| < 8GB | 32 | 8 | 50 | Low VRAM |
| CPU only | 16 | 4 | 20 | Minimal |

Use `--low-vram` flag to force low memory configuration:

```bash
# For GPUs with 6GB or less VRAM
python scripts/xai_experiments/run_xai_experiments.py --diet --low-vram
```

**Note for low VRAM GPUs (<8GB):**
- IMDB dataset max_length is automatically reduced from 256 to 128 tokens
- Text batch size is reduced to 8 for all datasets
- Gradient accumulation can be used for larger effective batch sizes

## 6. API Reference

### ComparisonConfig

```python
from scripts.xai_experiments import ComparisonConfig

config = ComparisonConfig(
    device="cuda",                    # Device for computation
    image_model_type="resnet",        # CNN architecture
    image_batch_size=64,              # Batch size for images
    image_epochs=5,                   # Training epochs
    image_max_samples=5000,           # Training samples
    image_comparison_samples=100,     # Samples for metric computation
    text_model_name="bert-base-uncased",
    text_batch_size=16,               # Batch size for text (reduce for low VRAM)
    text_max_length=128,              # Max sequence length (IMDB uses 256 by default)
    text_max_samples=2000,
    text_comparison_samples=50,
    diet_upsample_factor=4,           # DiET mask upsampling
    diet_rounding_steps=2,            # DiET distillation steps
    output_dir="./outputs"
)
```

### XAIMethodsComparison

```python
from scripts.xai_experiments import XAIMethodsComparison

comparison = XAIMethodsComparison(config)

# Run full comparison
results = comparison.run_full_comparison(
    run_images=True,
    run_text=True,
    skip_training=False
)

# Generate visualizations
comparison.visualize_results(save_plots=True, show=False)

# Get pandas DataFrame
df = comparison.get_results_dataframe()
```