#!/usr/bin/env python3
"""XAI Comparison Framework - Main Entry Point.

This script provides the main entry point for running DiET vs GradCAM/IG comparisons.
It can be used from command line or imported as a module for notebook usage.

Usage:
    Command line:
        python run_xai_experiments.py --diet              # Run full comparison
        python run_xai_experiments.py --diet --diet-images  # Images only
        python run_xai_experiments.py --diet --diet-text    # Text only
        python run_xai_experiments.py --diet --low-vram     # Low memory mode
    
    In notebook:
        from run_xai_experiments import run_comparison
        results = run_comparison(run_images=True, run_text=True)
"""

import os
import sys
import argparse
import json
import time
import torch
from typing import Dict, Any, Optional


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)


MIN_GLUE_BATCH_SIZE = 4
LOW_VRAM_THRESHOLD_GB = 8


def get_device_config(low_vram: bool = False) -> Dict[str, Any]:
    """Get device configuration optimized for available GPU.

    Args:
        low_vram: Whether to use low VRAM configuration

    Returns:
        Configuration dictionary
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Detected GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.1f} GB")

        if gpu_memory < LOW_VRAM_THRESHOLD_GB or low_vram:
            print("Using low VRAM configuration")
            return {
                "device": device,
                "batch_size": 16,
                "cnn_batch_size": 32,
                "glue_batch_size": 8,
                "num_workers": 2,
                "cnn_epochs": 5,
                "glue_epochs": 2,
                "sample_size": 5000,
                "ig_steps": 20,
                "gradcam_samples": 8,
                "comparison_samples": 50,
                "text_comparison_samples": 20,
            }
        else:
            print("Using standard configuration")
            return {
                "device": device,
                "batch_size": 64,
                "cnn_batch_size": 64,
                "glue_batch_size": 16,
                "num_workers": 4,
                "cnn_epochs": 10,
                "glue_epochs": 3,
                "sample_size": 10000,
                "ig_steps": 50,
                "gradcam_samples": 16,
                "comparison_samples": 100,
                "text_comparison_samples": 50,
            }
    else:
        print("No GPU detected, using CPU")
        return {
            "device": device,
            "batch_size": 16,
            "cnn_batch_size": 16,
            "glue_batch_size": MIN_GLUE_BATCH_SIZE,
            "num_workers": 0,
            "cnn_epochs": 2,
            "glue_epochs": 1,
            "sample_size": 2000,
            "ig_steps": 10,
            "gradcam_samples": 4,
            "comparison_samples": 20,
            "text_comparison_samples": 10,
        }


def run_cifar10_experiment(
    config: Dict[str, Any], args: argparse.Namespace
) -> Dict[str, Any]:
    """Run CIFAR-10 GradCAM experiment (deprecated - use --diet instead).

    This is now part of the DiET comparison framework.
    Use --diet --diet-images for DiET vs GradCAM comparison.

    Args:
        config: Device configuration
        args: Command line arguments

    Returns:
        Experiment results
    """
    print("\n" + "=" * 60)
    print("DEPRECATED: Use --diet --diet-images instead")
    print("Redirecting to DiET vs GradCAM comparison...")
    print("=" * 60)
    
    # Redirect to DiET comparison with images only
    args.diet_images = True
    args.diet_text = False
    return run_diet_experiment(config, args)


def run_glue_experiment(
    config: Dict[str, Any], args: argparse.Namespace
) -> Dict[str, Any]:
    """Run GLUE SST-2 BERT experiment (deprecated - use --diet instead).

    This is now part of the DiET comparison framework.
    Use --diet --diet-text for DiET vs IG comparison.

    Args:
        config: Device configuration
        args: Command line arguments

    Returns:
        Experiment results
    """
    print("\n" + "=" * 60)
    print("DEPRECATED: Use --diet --diet-text instead")
    print("Redirecting to DiET vs IG comparison...")
    print("=" * 60)
    
    # Redirect to DiET comparison with text only
    args.diet_images = False
    args.diet_text = True
    return run_diet_experiment(config, args)


def run_model_comparison(
    config: Dict[str, Any], args: argparse.Namespace
) -> Dict[str, Any]:
    """Run model comparison experiment (removed).

    This experiment has been removed as it is not related to
    DiET vs GradCAM/IG comparison. Use --diet for XAI comparison.

    Args:
        config: Device configuration
        args: Command line arguments

    Returns:
        Empty results
    """
    print("\n" + "=" * 60)
    print("REMOVED: Model comparison is no longer available")
    print("Use --diet for DiET vs GradCAM/IG comparison instead")
    print("=" * 60)
    
    return {"error": "Model comparison has been removed. Use --diet instead."}


def run_diet_experiment(
    config: Dict[str, Any], args: argparse.Namespace
) -> Dict[str, Any]:
    """Run DiET comparison experiment.

    Compares DiET discriminative feature attribution with basic XAI methods
    (GradCAM for images, Integrated Gradients for text).

    This is the main experiment function for the comparison framework.

    Args:
        config: Device configuration
        args: Command line arguments

    Returns:
        Comparison results with metrics and visualizations
    """
    from experiments.xai_comparison import XAIMethodsComparison, ComparisonConfig

    print("\n" + "=" * 60)
    print("DiET vs Basic XAI Methods Comparison Framework")
    print("=" * 60)
    print("Comparing:")
    print("  - Images: DiET vs GradCAM on CIFAR-10")
    print("  - Text: DiET vs Integrated Gradients on SST-2")
    print("=" * 60)

    # Build ComparisonConfig
    comparison_config = ComparisonConfig(
        device=config["device"],
        image_batch_size=config["cnn_batch_size"],
        image_epochs=min(5, config.get("cnn_epochs", 5)),
        image_max_samples=config.get("sample_size", 5000),
        image_comparison_samples=config.get("comparison_samples", 100),
        text_epochs=min(3, config.get("glue_epochs", 2)),
        text_max_samples=config.get("sample_size", 2000) // 2,
        text_comparison_samples=config.get("text_comparison_samples", 50),
        output_dir=os.path.join(args.output_dir, "diet_comparison"),
    )

    comparison = XAIMethodsComparison(comparison_config)

    run_images = args.diet_images or not args.diet_text
    run_text = args.diet_text or not args.diet_images

    results = comparison.run_full_comparison(
        run_images=run_images, run_text=run_text, skip_training=args.skip_training
    )
    
    # Generate visualizations
    if results:
        try:
            comparison.visualize_results(save_plots=True, show=False)
        except Exception as e:
            print(f"Visualization generation failed: {e}")

    return results


def run_comparison(
    run_images: bool = True,
    run_text: bool = True,
    low_vram: bool = False,
    output_dir: str = "./outputs/xai_experiments",
    skip_training: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """Convenience function for notebook usage.
    
    Run DiET comparison experiments with sensible defaults.
    
    Args:
        run_images: Whether to run image experiments (DiET vs GradCAM)
        run_text: Whether to run text experiments (DiET vs IG)
        low_vram: Use low memory configuration
        output_dir: Output directory for results
        skip_training: Skip training, use saved models
        **kwargs: Additional configuration overrides
        
    Returns:
        Comparison results dictionary
        
    Example:
        >>> from run_xai_experiments import run_comparison
        >>> results = run_comparison(run_images=True, run_text=False)
        >>> print(results["image_experiments"])
    """
    from experiments.xai_comparison import XAIMethodsComparison, ComparisonConfig
    
    config = get_device_config(low_vram)
    
    comparison_config = ComparisonConfig(
        device=config["device"],
        image_batch_size=config["cnn_batch_size"],
        image_epochs=config.get("cnn_epochs", 5),
        image_max_samples=config.get("sample_size", 5000),
        image_comparison_samples=config.get("comparison_samples", 100),
        text_epochs=config.get("glue_epochs", 2),
        text_max_samples=config.get("sample_size", 2000) // 2,
        text_comparison_samples=config.get("text_comparison_samples", 50),
        output_dir=os.path.join(output_dir, "diet_comparison"),
    )
    
    # Apply any overrides
    for key, value in kwargs.items():
        if hasattr(comparison_config, key):
            setattr(comparison_config, key, value)
    
    comparison = XAIMethodsComparison(comparison_config)
    
    results = comparison.run_full_comparison(
        run_images=run_images,
        run_text=run_text,
        skip_training=skip_training
    )
    
    # Generate visualizations
    if results:
        try:
            comparison.visualize_results(save_plots=True, show=False)
        except Exception as e:
            print(f"Visualization generation failed: {e}")
    
    return results


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="DiET vs Basic XAI Methods Comparison Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This framework compares DiET (Discriminative Feature Attribution) with:
  - GradCAM for image classification (CIFAR-10 dataset)
  - Integrated Gradients for text classification (SST-2 dataset)

Datasets:
  - CIFAR-10: 60,000 32x32 color images in 10 classes
  - SST-2: Stanford Sentiment Treebank for binary sentiment analysis

Examples:
  # Main use case: Run full DiET comparison (recommended)
  python run_xai_experiments.py --diet
  
  # Run only image comparison (DiET vs GradCAM on CIFAR-10)
  python run_xai_experiments.py --diet --diet-images
  
  # Run only text comparison (DiET vs IG on SST-2)
  python run_xai_experiments.py --diet --diet-text
  
  # Low VRAM mode for smaller GPUs
  python run_xai_experiments.py --diet --low-vram
  
  # Skip training, use saved models
  python run_xai_experiments.py --diet --skip-training
        """,
    )

    # Main comparison options
    parser.add_argument(
        "--diet", action="store_true", 
        help="Run DiET vs basic XAI comparison (main use case)"
    )
    parser.add_argument(
        "--diet-images",
        action="store_true",
        help="Run DiET comparison for images only (DiET vs GradCAM on CIFAR-10)",
    )
    parser.add_argument(
        "--diet-text",
        action="store_true",
        help="Run DiET comparison for text only (DiET vs IG on SST-2)",
    )
    
    # Legacy options (deprecated, redirect to --diet)
    parser.add_argument("--all", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--cifar10", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--glue", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--compare", action="store_true", help=argparse.SUPPRESS)

    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Data directory (default: ./data)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/xai_experiments",
        help="Output directory (default: ./outputs/xai_experiments)",
    )

    parser.add_argument(
        "--low-vram",
        action="store_true",
        help="Use low VRAM configuration for GPUs with < 8GB",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID (default: 0)")

    parser.add_argument("--epochs", type=int, help="Override number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training, use pretrained models",
    )

    parser.add_argument(
        "--model-type",
        type=str,
        default="resnet",
        choices=["simple", "resnet"],
        help="CNN model type (default: resnet)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Any of the flags should trigger the comparison
    should_run = args.diet or args.diet_images or args.diet_text or args.all or args.cifar10 or args.glue or args.compare
    
    if not should_run:
        print("=" * 60)
        print("DiET vs Basic XAI Methods Comparison Framework")
        print("=" * 60)
        print("\nDatasets:")
        print("  - CIFAR-10: For image classification (DiET vs GradCAM)")
        print("  - SST-2: For text classification (DiET vs Integrated Gradients)")
        print("\nNo experiment selected. Use --help for usage information.")
        print("\nQuick start (recommended):")
        print("  python run_xai_experiments.py --diet")
        return

    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print("=" * 60)
    print("DiET vs Basic XAI Methods Comparison Framework")
    print("=" * 60)
    print("\nDatasets:")
    print("  - CIFAR-10: For image classification (DiET vs GradCAM)")
    print("  - SST-2: For text classification (DiET vs Integrated Gradients)")

    config = get_device_config(args.low_vram)

    if args.epochs:
        config["cnn_epochs"] = args.epochs
        config["glue_epochs"] = args.epochs
    if args.batch_size:
        config["batch_size"] = args.batch_size
        config["cnn_batch_size"] = args.batch_size
        config["glue_batch_size"] = max(MIN_GLUE_BATCH_SIZE, args.batch_size // 4)

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {"config": config, "experiments": {}}

    total_start = time.time()

    try:
        # All paths now go through run_diet_experiment
        all_results["experiments"]["diet_comparison"] = run_diet_experiment(
            config, args
        )

    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
    except Exception as e:
        print(f"\n\nError during experiment: {e}")
        import traceback

        traceback.print_exc()

    total_time = time.time() - total_start
    all_results["total_time_seconds"] = total_time

    results_path = os.path.join(args.output_dir, "all_results.json")
    with open(results_path, "w") as f:

        def make_serializable(obj):
            if hasattr(obj, "tolist"):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, (int, float, bool, str, type(None))):
                return obj
            return str(obj)

        json.dump(make_serializable(all_results), f, indent=2)

    print("\n" + "=" * 60)
    print("Comparison Complete!")
    print("=" * 60)
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"Results saved to: {args.output_dir}")

    # Print diet comparison results (main focus)
    if "diet_comparison" in all_results["experiments"]:
        print("\n" + "-" * 40)
        print("DiET vs Basic XAI Methods - Summary")
        print("-" * 40)
        diet_results = all_results["experiments"]["diet_comparison"]
        if isinstance(diet_results, dict):
            if (
                "image_experiments" in diet_results
                and diet_results["image_experiments"]
            ):
                img = diet_results["image_experiments"]
                print("\n📸 Image Classification (CIFAR-10):")
                print(f"   Baseline Accuracy: {img.get('baseline_accuracy', 'N/A'):.2f}%")
                print(f"   GradCAM Score: {img.get('gradcam_mean_score', 'N/A'):.4f}")
                print(f"   DiET Score: {img.get('diet_mean_score', 'N/A'):.4f}")
                if img.get("diet_better"):
                    print(f"   ✓ DiET improves attribution by {img.get('improvement', 0):.4f}")
                else:
                    print(f"   → GradCAM sufficient for this task")
                    
            if "text_experiments" in diet_results and diet_results["text_experiments"]:
                txt = diet_results["text_experiments"]
                print("\n📝 Text Classification (SST-2):")
                print(f"   Baseline Accuracy: {txt.get('baseline_accuracy', 'N/A'):.2f}%")
                overlap = txt.get('ig_diet_overlap', 0)
                print(f"   IG-DiET Token Overlap: {overlap:.4f}")
                if overlap > 0.5:
                    print("   → High agreement between methods")
                else:
                    print("   → DiET identifies different discriminative features")

    print("\n" + "=" * 60)
    print("For notebook usage:")
    print("  from run_xai_experiments import run_comparison")
    print("  results = run_comparison(run_images=True, run_text=True)")
    print("=" * 60)


if __name__ == "__main__":
    main()
