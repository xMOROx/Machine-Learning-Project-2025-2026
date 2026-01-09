"""Experiments module for XAI comparison framework.

This module provides the core experiment classes for comparing
DiET with GradCAM (images) and Integrated Gradients (text).
"""

from .diet_experiment import DiETExperiment, DiETExplainer
from .diet_text_experiment import DiETTextExperiment, DiETTextExplainer
from .xai_comparison import XAIMethodsComparison, ComparisonConfig, run_diet_comparison

# Keep these for backward compatibility but they are not the main focus
from .cifar10_experiment import CIFAR10Experiment
from .glue_experiment import GLUEExperiment
from .model_comparison import ModelComparison

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
    # Backward compatibility
    "CIFAR10Experiment",
    "GLUEExperiment",
    "ModelComparison",
]
