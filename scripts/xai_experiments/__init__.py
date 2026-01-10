"""XAI Comparison Framework - DiET vs GradCAM/IG.

This package provides a comprehensive framework for comparing
Discriminative Feature Attribution (DiET) with basic XAI methods:

- **Images**: DiET vs GradCAM on CIFAR-10, CIFAR-100, SVHN, Fashion-MNIST
- **Text**: DiET vs Integrated Gradients on SST-2, IMDB, AG News

The framework includes:
- Robust evaluation metrics (Pixel Perturbation, AOPC, Faithfulness, etc.)
- Comprehensive visualizations
- Notebook-friendly API
- Resumable training via checkpoints (for laptop training)

Quick Start:
    >>> from xai_experiments import XAIMethodsComparison, ComparisonConfig
    >>> config = ComparisonConfig(device="cuda")
    >>> comparison = XAIMethodsComparison(config)
    >>> results = comparison.run_full_comparison()
    >>> comparison.visualize_results()

For more details, see the experiments module.
"""

__version__ = "0.3.0"

# Main exports for notebook usage
from .experiments.xai_comparison import (
    XAIMethodsComparison,
    ComparisonConfig,
    run_diet_comparison,
)

# Checkpoint manager for resumable training
from .utils.checkpointing import CheckpointManager, get_default_checkpoint_manager

__all__ = [
    "XAIMethodsComparison",
    "ComparisonConfig",
    "run_diet_comparison",
    "CheckpointManager",
    "get_default_checkpoint_manager",
]
