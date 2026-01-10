"""Experiments module for XAI comparison framework.

This module provides the core experiment classes for comparing
DiET with GradCAM (images) and Integrated Gradients (text).

Datasets used:
- CIFAR-10: For image classification and DiET vs GradCAM comparison
- SST-2 (GLUE): For text classification and DiET vs Integrated Gradients comparison
"""

from .diet_experiment import DiETExperiment, DiETExplainer
from .diet_text_experiment import DiETTextExperiment, DiETTextExplainer
from .xai_comparison import XAIMethodsComparison, ComparisonConfig, run_diet_comparison

__all__ = [
    # Main comparison framework
    "XAIMethodsComparison",
    "ComparisonConfig",
    "run_diet_comparison",
    # DiET experiments
    "DiETExperiment",
    "DiETExplainer",
    "DiETTextExperiment",
    "DiETTextExplainer",
]
