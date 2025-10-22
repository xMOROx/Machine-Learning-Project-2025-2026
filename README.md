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

This project builds directly on the official code from the authors. We will clone them as sub-modules.
