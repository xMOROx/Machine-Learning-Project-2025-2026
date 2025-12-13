"""XAI Methods Comparison Module.

This module provides comprehensive comparison between:
- Basic XAI methods: GradCAM, Integrated Gradients
- DiET: Discriminative Feature Attribution

Evaluates and compares:
1. Attribution quality (pixel perturbation)
2. Faithfulness to model
3. Localization accuracy
4. Computational cost
"""

import os
import time
import json
import traceback
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime

# Use non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class XAIMethodsComparison:
    """Comprehensive comparison of XAI methods.
    
    Compares:
    - GradCAM (image baseline)
    - Integrated Gradients (image and text)
    - DiET (discriminative attribution for images and text)
    
    Metrics:
    - Pixel perturbation accuracy (images)
    - Top-k token overlap (text)
    - Faithfulness scores
    - Time complexity
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize comparison module.
        
        Args:
            config: Configuration with:
                - output_dir: Where to save results
                - device: CUDA/CPU
                - run_images: Whether to run image experiments
                - run_text: Whether to run text experiments
        """
        self.config = config
        self.output_dir = config.get("output_dir", "./outputs/xai_experiments/comparison")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "image_experiments": {},
            "text_experiments": {},
            "summary": {}
        }
    
    def run_image_comparison(self, skip_training: bool = False) -> Dict[str, Any]:
        """Run image-based XAI comparison (CIFAR-10).
        
        Compares GradCAM vs DiET.
        
        Args:
            skip_training: If True, try to load previously trained models
        
        Returns:
            Image experiment results
        """
        print("\n" + "=" * 70)
        print("IMAGE-BASED XAI COMPARISON (CIFAR-10)")
        print("=" * 70)
        
        from .diet_experiment import DiETExperiment
        
        # Configure for RTX 3060 (12GB VRAM)
        image_config = {
            "device": self.config.get("device", "cuda"),
            "data_dir": self.config.get("data_dir", "./data"),
            "output_dir": os.path.join(self.output_dir, "cifar10"),
            "model_type": "resnet",
            "batch_size": self.config.get("batch_size", 64),
            "max_samples": self.config.get("max_samples_image", 3000),
            "baseline_epochs": self.config.get("epochs_image", 3),
            "comparison_samples": self.config.get("comparison_samples", 16),
            "upsample_factor": 4,
            "rounding_steps": 2
        }
        
        # Run DiET experiment (includes GradCAM comparison)
        experiment = DiETExperiment(image_config)
        results = experiment.run_full_experiment(skip_training=skip_training)
        
        self.results["image_experiments"] = {
            "baseline_accuracy": results["baseline"]["final_test_acc"],
            "diet_accuracy": results["diet"]["final_test_acc"],
            "gradcam_perturbation": results["comparison"]["gradcam"]["perturbation_scores"],
            "diet_perturbation": results["comparison"]["diet"]["perturbation_scores"],
            "gradcam_mean_score": results["comparison"]["gradcam"]["mean_score"],
            "diet_mean_score": results["comparison"]["diet"]["mean_score"],
            "improvement": results["comparison"]["improvement"],
            "diet_better": results["comparison"]["improvement"] > 0
        }
        
        return self.results["image_experiments"]
    
    def run_text_comparison(self, skip_training: bool = False) -> Dict[str, Any]:
        """Run text-based XAI comparison (SST-2).
        
        Compares Integrated Gradients vs DiET.
        
        Args:
            skip_training: If True, try to load previously trained models
        
        Returns:
            Text experiment results
        """
        print("\n" + "=" * 70)
        print("TEXT-BASED XAI COMPARISON (SST-2)")
        print("=" * 70)
        
        from .diet_text_experiment import DiETTextExperiment
        
        text_config = {
            "device": self.config.get("device", "cuda"),
            "data_dir": self.config.get("data_dir", "./data"),
            "output_dir": os.path.join(self.output_dir, "sst2"),
            "model_name": "bert-base-uncased",
            "max_length": 128,
            "max_samples": self.config.get("max_samples_text", 1000),
            "epochs": self.config.get("epochs_text", 2),
            "comparison_samples": self.config.get("comparison_samples_text", 10),
            "rounding_steps": 2
        }
        
        experiment = DiETTextExperiment(text_config)
        results = experiment.run_full_experiment(skip_training=skip_training)
        
        self.results["text_experiments"] = {
            "baseline_accuracy": results["baseline"]["val_acc"],
            "ig_diet_overlap": results["comparison"]["mean_top_k_overlap"],
            "samples_compared": results["comparison"]["num_samples"]
        }
        
        return self.results["text_experiments"]
    
    def generate_summary_report(self) -> str:
        """Generate comprehensive comparison report.
        
        Returns:
            Report as formatted string
        """
        report = []
        report.append("=" * 70)
        report.append("XAI METHODS COMPARISON REPORT")
        report.append("=" * 70)
        report.append(f"\nGenerated: {self.results['timestamp']}")
        report.append(f"Device: {self.config.get('device', 'cuda')}")
        
        # Image experiments
        if self.results["image_experiments"]:
            img = self.results["image_experiments"]
            report.append("\n" + "-" * 50)
            report.append("IMAGE CLASSIFICATION (CIFAR-10)")
            report.append("-" * 50)
            report.append(f"\nModel Accuracy:")
            report.append(f"  Baseline: {img['baseline_accuracy']:.2f}%")
            report.append(f"  After DiET: {img['diet_accuracy']:.2f}%")
            
            report.append(f"\nPixel Perturbation Results:")
            report.append(f"  GradCAM Mean Score: {img['gradcam_mean_score']:.4f}")
            report.append(f"  DiET Mean Score: {img['diet_mean_score']:.4f}")
            
            if img["diet_better"]:
                report.append(f"\n  ✓ DiET IMPROVES attribution by {img['improvement']:.4f}")
                report.append("  → DiET masks focus more on discriminative features")
            else:
                report.append(f"\n  → GradCAM performs better by {-img['improvement']:.4f}")
                report.append("  → Basic gradient methods suffice for this task")
        
        # Text experiments
        if self.results["text_experiments"]:
            txt = self.results["text_experiments"]
            report.append("\n" + "-" * 50)
            report.append("TEXT CLASSIFICATION (SST-2)")
            report.append("-" * 50)
            report.append(f"\nModel Accuracy:")
            report.append(f"  BERT Baseline: {txt['baseline_accuracy']:.2f}%")
            
            report.append(f"\nToken Attribution Comparison:")
            report.append(f"  IG-DiET Top-k Overlap: {txt['ig_diet_overlap']:.4f}")
            report.append(f"  Samples Compared: {txt['samples_compared']}")
            
            if txt["ig_diet_overlap"] > 0.5:
                report.append("\n  → High agreement between methods")
                report.append("  → Both identify similar important tokens")
            else:
                report.append("\n  → Methods identify different important tokens")
                report.append("  → DiET may capture discriminative features IG misses")
        
        # Overall summary
        report.append("\n" + "=" * 70)
        report.append("CONCLUSIONS")
        report.append("=" * 70)
        
        if self.results["image_experiments"] and self.results["image_experiments"].get("diet_better"):
            report.append("\n• For IMAGES: DiET provides more discriminative attributions")
            report.append("  - Masks learned by DiET better preserve prediction-relevant features")
            report.append("  - Recommended for applications requiring faithful explanations")
        else:
            report.append("\n• For IMAGES: GradCAM remains effective for quick explanations")
            report.append("  - Gradient-based methods are computationally efficient")
            report.append("  - DiET requires additional training but may not improve results")
        
        if self.results["text_experiments"]:
            overlap = self.results["text_experiments"].get("ig_diet_overlap", 0)
            if overlap < 0.3:
                report.append("\n• For TEXT: DiET captures unique discriminative patterns")
                report.append("  - Low overlap suggests DiET finds different important tokens")
                report.append("  - May be valuable for sentiment-specific attributions")
            else:
                report.append("\n• For TEXT: IG and DiET largely agree on important tokens")
                report.append("  - Standard IG may be sufficient for most applications")
        
        report.append("\n" + "=" * 70)
        
        self.results["summary"]["report"] = "\n".join(report)
        
        return self.results["summary"]["report"]
    
    def save_results(self) -> str:
        """Save all results to files.
        
        Returns:
            Path to results directory
        """
        # Save JSON results
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
        
        # Save text report
        report_path = os.path.join(self.output_dir, "comparison_report.txt")
        with open(report_path, "w") as f:
            f.write(self.results["summary"].get("report", "No report generated"))
        
        # Create summary visualization
        self._create_summary_visualization()
        
        print(f"\nResults saved to: {self.output_dir}")
        
        return self.output_dir
    
    def _create_summary_visualization(self) -> None:
        """Create summary visualization charts."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Image comparison
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
        
        # Text comparison
        if self.results["text_experiments"]:
            txt = self.results["text_experiments"]
            
            # Overlap visualization
            overlap = txt["ig_diet_overlap"]
            axes[1].barh(["IG-DiET\nAgreement"], [overlap], color="#FF9800")
            axes[1].set_xlim(0, 1)
            axes[1].set_title("Text Attribution Comparison\n(Top-k Token Overlap)")
            axes[1].text(overlap + 0.02, 0, f"{overlap:.3f}", va="center", fontweight="bold")
        else:
            axes[1].text(0.5, 0.5, "No text experiments run", ha="center", va="center")
            axes[1].set_title("Text Attribution Comparison")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "comparison_summary.png"), dpi=150, bbox_inches="tight")
        plt.close()
    
    def run_full_comparison(
        self,
        run_images: bool = True,
        run_text: bool = True,
        skip_training: bool = False
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
        
        # Run experiments
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
        
        # Generate report
        report = self.generate_summary_report()
        print(report)
        
        # Save results
        self.save_results()
        
        total_time = time.time() - total_start
        self.results["total_time_seconds"] = total_time
        
        print(f"\nTotal comparison time: {total_time/60:.1f} minutes")
        
        return self.results


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
        "comparison_samples_text": 10
    }
    
    if config:
        default_config.update(config)
    
    comparison = XAIMethodsComparison(default_config)
    return comparison.run_full_comparison()
