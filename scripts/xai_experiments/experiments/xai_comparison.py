"""XAI Methods Comparison Module - DiET vs GradCAM/IG Framework.

This module provides a comprehensive framework for comparing:
- DiET (Discriminative Feature Attribution) with GradCAM for images
- DiET with Integrated Gradients for text

This is designed as a notebook-friendly framework with:
1. Rich metrics (Pixel Perturbation, AOPC, Insertion/Deletion, Faithfulness)
2. Comprehensive visualizations
3. Easy-to-use API for Jupyter notebooks
4. Robust evaluation on larger datasets

Reference:
- DiET: Bhalla et al., "Discriminative Feature Attributions", NeurIPS 2023
"""

import os
import time
import json
import traceback
import numpy as np
import torch
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass, field


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class ComparisonConfig:
    """Configuration for XAI comparison experiments."""
    # Device settings
    device: str = "cuda"
    
    # Image experiment settings
    image_dataset: str = "cifar10"
    image_model_type: str = "resnet"
    image_batch_size: int = 64
    image_epochs: int = 5
    image_max_samples: int = 5000
    image_comparison_samples: int = 100
    
    # Text experiment settings
    text_dataset: str = "sst2"
    text_model_name: str = "bert-base-uncased"
    text_max_length: int = 128
    text_max_samples: int = 2000
    text_epochs: int = 2
    text_comparison_samples: int = 50
    
    # DiET settings
    diet_upsample_factor: int = 4
    diet_rounding_steps: int = 2
    
    # Metric settings
    perturbation_percentages: List[int] = field(default_factory=lambda: [5, 10, 20, 30, 50, 70, 90])
    insertion_deletion_steps: int = 50
    aopc_steps: int = 10
    faithfulness_samples: int = 30
    
    # Output settings
    output_dir: str = "./outputs/xai_experiments/comparison"
    save_visualizations: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            k: v if not isinstance(v, list) else v
            for k, v in self.__dict__.items()
        }


class XAIMethodsComparison:
    """Comprehensive comparison of XAI methods - DiET vs GradCAM/IG.

    This is the main framework class for comparing attribution methods.
    Designed to be used both as a standalone module and in Jupyter notebooks.

    Compares:
    - GradCAM vs DiET for image classification (CIFAR-10)
    - Integrated Gradients vs DiET for text classification (SST-2)

    Metrics computed:
    - Pixel Perturbation (keep/remove important pixels)
    - AOPC (Area Over Perturbation Curve)
    - Insertion/Deletion curves
    - Faithfulness Correlation
    - Top-k Token Overlap (for text)

    Example usage in notebook:
        >>> from experiments.xai_comparison import XAIMethodsComparison, ComparisonConfig
        >>> config = ComparisonConfig(device="cuda", image_comparison_samples=50)
        >>> comparison = XAIMethodsComparison(config)
        >>> results = comparison.run_full_comparison()
        >>> comparison.visualize_results()
    """

    def __init__(self, config: Optional[ComparisonConfig] = None):
        """Initialize comparison module.

        Args:
            config: ComparisonConfig object or None for defaults.
                   Can also pass a dictionary for backward compatibility.
        """
        if config is None:
            self.config = ComparisonConfig()
        elif isinstance(config, dict):
            # Backward compatibility with dict config
            self.config = ComparisonConfig()
            for key, value in config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                elif key == "output_dir":
                    self.config.output_dir = value
        else:
            self.config = config
        
        self.output_dir = self.config.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.results = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config.to_dict() if hasattr(self.config, 'to_dict') else self.config,
            "image_experiments": {},
            "text_experiments": {},
            "metrics": {},
        }
        
        # Initialize visualizer lazily
        self._visualizer = None
        
        # Compute data directory path
        self._data_dir = self._compute_data_dir()
    
    def _compute_data_dir(self) -> str:
        """Compute the data directory path."""
        # Default data dir is relative to output_dir parent
        output_parent = os.path.dirname(os.path.dirname(self.output_dir))
        if output_parent and os.path.isdir(output_parent):
            return os.path.join(output_parent, "data")
        return "./data"
    
    def _get_config_value(self, attr_name: str, default_value):
        """Safely get a config value with fallback to default."""
        if hasattr(self.config, attr_name):
            return getattr(self.config, attr_name)
        return default_value

    def run_image_comparison(self, skip_training: bool = False) -> Dict[str, Any]:
        """Run image-based XAI comparison (CIFAR-10).

        Compares GradCAM vs DiET with comprehensive metrics.

        Args:
            skip_training: If True, try to load previously trained models

        Returns:
            Image experiment results with detailed metrics
        """
        print("\n" + "=" * 70)
        print("IMAGE-BASED XAI COMPARISON (CIFAR-10)")
        print("DiET vs GradCAM - Discriminative Feature Attribution")
        print("=" * 70)

        from .diet_experiment import DiETExperiment

        # Build config using helper method for clean access
        image_config = {
            "device": self._get_config_value("device", "cuda"),
            "data_dir": self._data_dir,
            "output_dir": os.path.join(self.output_dir, "cifar10"),
            "model_type": self._get_config_value("image_model_type", "resnet"),
            "batch_size": self._get_config_value("image_batch_size", 64),
            "max_samples": self._get_config_value("image_max_samples", 3000),
            "baseline_epochs": self._get_config_value("image_epochs", 3),
            "comparison_samples": self._get_config_value("image_comparison_samples", 16),
            "upsample_factor": self._get_config_value("diet_upsample_factor", 4),
            "rounding_steps": self._get_config_value("diet_rounding_steps", 2),
        }

        experiment = DiETExperiment(image_config)
        results = experiment.run_full_experiment(skip_training=skip_training)

        self.results["image_experiments"] = {
            "baseline_accuracy": results["baseline"]["final_test_acc"],
            "diet_accuracy": results["diet"]["final_test_acc"],
            "gradcam_perturbation": results["comparison"]["gradcam"][
                "perturbation_scores"
            ],
            "diet_perturbation": results["comparison"]["diet"]["perturbation_scores"],
            "gradcam_mean_score": results["comparison"]["gradcam"]["mean_score"],
            "diet_mean_score": results["comparison"]["diet"]["mean_score"],
            "improvement": results["comparison"]["improvement"],
            "diet_better": results["comparison"]["improvement"] > 0,
        }

        return self.results["image_experiments"]

    def run_text_comparison(self, skip_training: bool = False) -> Dict[str, Any]:
        """Run text-based XAI comparison (SST-2).

        Compares Integrated Gradients vs DiET with token-level analysis.

        Args:
            skip_training: If True, try to load previously trained models

        Returns:
            Text experiment results with detailed metrics
        """
        print("\n" + "=" * 70)
        print("TEXT-BASED XAI COMPARISON (SST-2)")
        print("DiET vs Integrated Gradients - Token Attribution")
        print("=" * 70)

        from .diet_text_experiment import DiETTextExperiment

        # Build config using helper method for clean access
        text_config = {
            "device": self._get_config_value("device", "cuda"),
            "data_dir": self._data_dir,
            "output_dir": os.path.join(self.output_dir, "sst2"),
            "model_name": self._get_config_value("text_model_name", "bert-base-uncased"),
            "max_length": self._get_config_value("text_max_length", 128),
            "max_samples": self._get_config_value("text_max_samples", 1000),
            "epochs": self._get_config_value("text_epochs", 2),
            "comparison_samples": self._get_config_value("text_comparison_samples", 10),
            "rounding_steps": self._get_config_value("diet_rounding_steps", 2),
        }

        experiment = DiETTextExperiment(text_config)
        results = experiment.run_full_experiment(skip_training=skip_training)

        self.results["text_experiments"] = {
            "baseline_accuracy": results["baseline"]["val_acc"],
            "ig_diet_overlap": results["comparison"]["mean_top_k_overlap"],
            "samples_compared": results["comparison"]["num_samples"],
        }

        return self.results["text_experiments"]

    def generate_summary_report(self) -> str:
        """Generate comprehensive comparison report.

        Returns:
            Report as formatted string
        """
        device = self.config.device if hasattr(self.config, 'device') else self.config.get('device', 'cuda')
        
        report = []
        report.append("=" * 70)
        report.append("XAI METHODS COMPARISON REPORT")
        report.append("DiET vs GradCAM (Images) & DiET vs IG (Text)")
        report.append("=" * 70)
        report.append(f"\nGenerated: {self.results['timestamp']}")
        report.append(f"Device: {device}")

        if self.results["image_experiments"]:
            img = self.results["image_experiments"]
            report.append("\n" + "-" * 50)
            report.append("IMAGE CLASSIFICATION (CIFAR-10)")
            report.append("-" * 50)
            report.append("\nModel Accuracy:")
            report.append(f"  Baseline: {img['baseline_accuracy']:.2f}%")
            report.append(f"  After DiET: {img['diet_accuracy']:.2f}%")

            report.append("\nPixel Perturbation Results:")
            report.append(f"  GradCAM Mean Score: {img['gradcam_mean_score']:.4f}")
            report.append(f"  DiET Mean Score: {img['diet_mean_score']:.4f}")

            if img["diet_better"]:
                report.append(
                    f"\n  ✓ DiET IMPROVES attribution by {img['improvement']:.4f}"
                )
                report.append("  → DiET masks focus more on discriminative features")
            else:
                report.append(
                    f"\n  → GradCAM performs better by {-img['improvement']:.4f}"
                )
                report.append("  → Basic gradient methods suffice for this task")

        if self.results["text_experiments"]:
            txt = self.results["text_experiments"]
            report.append("\n" + "-" * 50)
            report.append("TEXT CLASSIFICATION (SST-2)")
            report.append("-" * 50)
            report.append("\nModel Accuracy:")
            report.append(f"  BERT Baseline: {txt['baseline_accuracy']:.2f}%")

            report.append("\nToken Attribution Comparison:")
            report.append(f"  IG-DiET Top-k Overlap: {txt['ig_diet_overlap']:.4f}")
            report.append(f"  Samples Compared: {txt['samples_compared']}")

            if txt["ig_diet_overlap"] > 0.5:
                report.append("\n  → High agreement between methods")
                report.append("  → Both identify similar important tokens")
            else:
                report.append("\n  → Methods identify different important tokens")
                report.append("  → DiET may capture discriminative features IG misses")
        
        # Store report in results
        report_str = "\n".join(report)
        self.results["summary"] = {"report": report_str}
        
        return report_str

    def save_results(self) -> str:
        """Save all results to files.

        Returns:
            Path to results directory
        """

        results_path = os.path.join(self.output_dir, "comparison_results.json")

        def make_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (bool, np.bool_)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            return obj

        with open(results_path, "w") as f:
            json.dump(make_serializable(self.results), f, indent=2)

        report_path = os.path.join(self.output_dir, "comparison_report.txt")
        with open(report_path, "w") as f:
            f.write(self.results["summary"].get("report", "No report generated"))

        self._create_summary_visualization()

        print(f"\nResults saved to: {self.output_dir}")

        return self.output_dir

    def _create_summary_visualization(self) -> None:
        """Create summary visualization charts."""
        _, axes = plt.subplots(1, 2, figsize=(14, 5))

        if self.results["image_experiments"]:
            img = self.results["image_experiments"]

            methods = ["GradCAM", "DiET"]
            scores = [img["gradcam_mean_score"], img["diet_mean_score"]]
            colors = ["#2196F3", "#4CAF50"]

            axes[0].bar(methods, scores, color=colors)
            axes[0].set_ylabel("Pixel Perturbation Score")
            axes[0].set_title("Image Attribution Quality\n(Higher = Better)")
            axes[0].set_ylim(0, 1)

            for i, v in enumerate(scores):
                axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")
        else:
            axes[0].text(0.5, 0.5, "No image experiments run", ha="center", va="center")
            axes[0].set_title("Image Attribution Quality")

        if self.results["text_experiments"]:
            txt = self.results["text_experiments"]

            overlap = txt["ig_diet_overlap"]
            axes[1].barh(["IG-DiET\nAgreement"], [overlap], color="#FF9800")
            axes[1].set_xlim(0, 1)
            axes[1].set_title("Text Attribution Comparison\n(Top-k Token Overlap)")
            axes[1].text(
                overlap + 0.02, 0, f"{overlap:.3f}", va="center", fontweight="bold"
            )
        else:
            axes[1].text(0.5, 0.5, "No text experiments run", ha="center", va="center")
            axes[1].set_title("Text Attribution Comparison")

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "comparison_summary.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    def run_full_comparison(
        self,
        run_images: bool = True,
        run_text: bool = True,
        skip_training: bool = False,
    ) -> Dict[str, Any]:
        """Run complete comparison pipeline.

        Args:
            run_images: Whether to run image experiments
            run_text: Whether to run text experiments
            skip_training: If True, try to load previously trained models

        Returns:
            All comparison results
        """
        print("=" * 70)
        print("XAI METHODS COMPARISON PIPELINE")
        print("DiET vs Basic XAI Methods (GradCAM, Integrated Gradients)")
        if skip_training:
            print("(Using previously trained models if available)")
        print("=" * 70)

        total_start = time.time()

        if run_images:
            try:
                self.run_image_comparison(skip_training=skip_training)
            except Exception as e:
                print(f"Image comparison failed: {e}")
                traceback.print_exc()

        if run_text:
            try:
                self.run_text_comparison(skip_training=skip_training)
            except Exception as e:
                print(f"Text comparison failed: {e}")
                traceback.print_exc()

        report = self.generate_summary_report()
        print(report)

        self.save_results()

        total_time = time.time() - total_start
        self.results["total_time_seconds"] = total_time

        print(f"\nTotal comparison time: {total_time / 60:.1f} minutes")

        return self.results
    
    def visualize_results(
        self,
        save_plots: bool = True,
        show: bool = False
    ) -> Dict[str, Any]:
        """Generate comprehensive visualizations of comparison results.
        
        This method creates all visualizations for the comparison results,
        including bar charts, radar plots, and HTML reports.
        
        Args:
            save_plots: Whether to save plots to disk
            show: Whether to display plots (for notebooks)
            
        Returns:
            Dictionary with paths to generated visualizations
        """
        try:
            from ..visualization import ComparisonVisualizer
        except ImportError:
            # Fallback to basic visualization
            print("Advanced visualization not available, using basic plots")
            self._create_summary_visualization()
            return {"summary_plot": os.path.join(self.output_dir, "comparison_summary.png")}
        
        visualizer = ComparisonVisualizer(output_dir=self.output_dir)
        generated_files = {}
        
        # Prepare metric results for plotting
        if self.results["image_experiments"]:
            img = self.results["image_experiments"]
            image_metrics = {
                "GradCAM": {"perturbation_score": img.get("gradcam_mean_score", 0)},
                "DiET": {"perturbation_score": img.get("diet_mean_score", 0)}
            }
            
            # Bar chart comparison
            fig = visualizer.plot_metric_comparison_bar(
                image_metrics,
                title="Image Attribution Quality (CIFAR-10): DiET vs GradCAM",
                save_name="image_metric_comparison",
                show=show
            )
            generated_files["image_bar_chart"] = os.path.join(
                self.output_dir, "image_metric_comparison.png"
            )
            plt.close(fig)
            
            # Perturbation curve if available
            if "gradcam_perturbation" in img and "diet_perturbation" in img:
                perturbation_results = {
                    "GradCAM": img["gradcam_perturbation"],
                    "DiET": img["diet_perturbation"]
                }
                fig = visualizer.plot_perturbation_curve(
                    perturbation_results,
                    title="Pixel Perturbation Analysis",
                    save_name="perturbation_curves",
                    show=show
                )
                generated_files["perturbation_curves"] = os.path.join(
                    self.output_dir, "perturbation_curves.png"
                )
                plt.close(fig)
        
        # Create summary dashboard
        fig = visualizer.create_summary_dashboard(
            image_results=self.results.get("image_experiments"),
            text_results=self.results.get("text_experiments"),
            save_name="comparison_dashboard",
            show=show
        )
        generated_files["dashboard"] = os.path.join(
            self.output_dir, "comparison_dashboard.png"
        )
        plt.close(fig)
        
        # Generate HTML report
        html_path = visualizer.generate_html_report(
            self.results,
            report_title="DiET vs Basic XAI Methods Comparison",
            save_name="comparison_report"
        )
        generated_files["html_report"] = html_path
        
        print(f"\nVisualization files generated:")
        for name, path in generated_files.items():
            print(f"  - {name}: {path}")
        
        return generated_files
    
    def get_results_dataframe(self):
        """Get results as a pandas DataFrame for easy analysis in notebooks.
        
        Returns:
            pandas DataFrame with comparison results
        """
        import pandas as pd
        
        data = []
        
        if self.results["image_experiments"]:
            img = self.results["image_experiments"]
            data.append({
                "Modality": "Image (CIFAR-10)",
                "Method 1": "GradCAM",
                "Method 2": "DiET",
                "GradCAM Score": img.get("gradcam_mean_score", 0),
                "DiET Score": img.get("diet_mean_score", 0),
                "Improvement": img.get("improvement", 0),
                "DiET Better": img.get("diet_better", False),
                "Baseline Accuracy": img.get("baseline_accuracy", 0),
                "DiET Accuracy": img.get("diet_accuracy", 0),
            })
        
        if self.results["text_experiments"]:
            txt = self.results["text_experiments"]
            data.append({
                "Modality": "Text (SST-2)",
                "Method 1": "IG",
                "Method 2": "DiET",
                "IG-DiET Overlap": txt.get("ig_diet_overlap", 0),
                "Samples Compared": txt.get("samples_compared", 0),
                "Baseline Accuracy": txt.get("baseline_accuracy", 0),
            })
        
        return pd.DataFrame(data)


def run_diet_comparison(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convenience function to run DiET comparison.

    Args:
        config: Optional configuration override

    Returns:
        Comparison results
    """
    default_config = {
        "device": "cuda",
        "output_dir": "./outputs/xai_experiments/diet_comparison",
        "data_dir": "./data",
        "batch_size": 32,
        "max_samples_image": 2000,
        "max_samples_text": 1000,
        "epochs_image": 3,
        "epochs_text": 2,
        "comparison_samples": 16,
        "comparison_samples_text": 10,
    }

    if config:
        default_config.update(config)

    comparison = XAIMethodsComparison(default_config)
    return comparison.run_full_comparison()
