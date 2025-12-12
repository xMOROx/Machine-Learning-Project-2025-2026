#!/usr/bin/env python3
"""XAI Experiments Orchestration Script.

This script orchestrates all XAI experiments including:
- CIFAR-10 classification with GradCAM
- GLUE SST-2 with BERT and Integrated Gradients
- Model comparison (CNN, RF, LightGBM, SVM, Logistic Regression)
- DiET comparison (discriminative feature attribution vs basic XAI)

Optimized for RTX 3060 (12GB VRAM) with configurable memory usage.

Usage:
    python run_xai_experiments.py --all                    # Run all experiments
    python run_xai_experiments.py --cifar10                # Run CIFAR-10 only
    python run_xai_experiments.py --glue                   # Run GLUE SST-2 only
    python run_xai_experiments.py --compare                # Run model comparison
    python run_xai_experiments.py --diet                   # Run DiET comparison
    python run_xai_experiments.py --low-vram               # Use low VRAM settings
"""

import os
import sys
import argparse
import json
import time
import torch
from typing import Dict, Any


# Add parent directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Configuration constants
MIN_GLUE_BATCH_SIZE = 4  # Minimum batch size for GLUE/BERT experiments
LOW_VRAM_THRESHOLD_GB = 8  # GPU memory threshold for low VRAM mode


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
        
        # RTX 3060 has 12GB VRAM
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
                "gradcam_samples": 8
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
                "gradcam_samples": 16
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
            "gradcam_samples": 4
        }


def run_cifar10_experiment(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Run CIFAR-10 GradCAM experiment.
    
    Args:
        config: Device configuration
        args: Command line arguments
        
    Returns:
        Experiment results
    """
    from experiments.cifar10_experiment import CIFAR10Experiment
    
    print("\n" + "=" * 60)
    print("Starting CIFAR-10 GradCAM Experiment")
    print("=" * 60)
    
    experiment_config = {
        "data_dir": args.data_dir,
        "output_dir": os.path.join(args.output_dir, "cifar10"),
        "model_type": args.model_type,
        "batch_size": config["cnn_batch_size"],
        "num_workers": config["num_workers"],
        "epochs": config["cnn_epochs"],
        "learning_rate": 0.001,
        "device": config["device"]
    }
    
    experiment = CIFAR10Experiment(experiment_config)
    
    if args.skip_training:
        # Load pretrained model if available
        model_path = os.path.join(experiment_config["output_dir"], f"{args.model_type}_cifar10.pth")
        if os.path.exists(model_path):
            experiment.prepare_data()
            experiment.load_model(model_path)
            results = experiment.generate_gradcam(num_samples=config["gradcam_samples"])
        else:
            print(f"Model not found at {model_path}. Running full pipeline.")
            results = experiment.run_full_pipeline()
    else:
        results = experiment.run_full_pipeline()
    
    return results


def run_glue_experiment(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Run GLUE SST-2 BERT Integrated Gradients experiment.
    
    Args:
        config: Device configuration
        args: Command line arguments
        
    Returns:
        Experiment results
    """
    from experiments.glue_experiment import GLUEExperiment
    
    print("\n" + "=" * 60)
    print("Starting GLUE SST-2 BERT Integrated Gradients Experiment")
    print("=" * 60)
    
    experiment_config = {
        "data_dir": args.data_dir,
        "output_dir": os.path.join(args.output_dir, "glue_sst2"),
        "model_name": "bert-base-uncased",
        "batch_size": config["glue_batch_size"],
        "max_length": 128,
        "epochs": config["glue_epochs"],
        "learning_rate": 2e-5,
        "device": config["device"]
    }
    
    experiment = GLUEExperiment(experiment_config)
    
    if args.skip_training:
        model_path = os.path.join(experiment_config["output_dir"], "bert_sst2")
        if os.path.exists(model_path):
            experiment.prepare_data()
            experiment.load_model(model_path)
            results = experiment.generate_integrated_gradients(
                num_samples=10,
                n_steps=config["ig_steps"]
            )
        else:
            print(f"Model not found at {model_path}. Running full pipeline.")
            results = experiment.run_full_pipeline()
    else:
        results = experiment.run_full_pipeline()
    
    return results


def run_model_comparison(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Run model comparison experiment.
    
    Args:
        config: Device configuration
        args: Command line arguments
        
    Returns:
        Comparison results
    """
    from experiments.model_comparison import ModelComparison
    
    print("\n" + "=" * 60)
    print("Starting Model Comparison Experiment")
    print("=" * 60)
    
    comparison_config = {
        "data_dir": args.data_dir,
        "output_dir": os.path.join(args.output_dir, "model_comparison"),
        "batch_size": config["cnn_batch_size"],
        "sample_size": config["sample_size"],
        "device": config["device"],
        "cnn_epochs": min(5, config["cnn_epochs"]),
        "cnn_learning_rate": 0.001,
        "rf_n_estimators": 100,
        "rf_max_depth": 20,
        "lgb_n_estimators": 100,
        "svm_kernel": "rbf",
        "lr_max_iter": 1000
    }
    
    comparison = ModelComparison(comparison_config)
    results = comparison.run_comparison()
    
    return results


def run_diet_experiment(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Run DiET comparison experiment.
    
    Compares DiET discriminative feature attribution with basic XAI methods
    (GradCAM for images, Integrated Gradients for text).
    
    Args:
        config: Device configuration
        args: Command line arguments
        
    Returns:
        Comparison results
    """
    from experiments.xai_comparison import XAIMethodsComparison
    
    print("\n" + "=" * 60)
    print("Starting DiET vs Basic XAI Comparison")
    print("=" * 60)
    
    comparison_config = {
        "device": config["device"],
        "data_dir": args.data_dir,
        "output_dir": os.path.join(args.output_dir, "diet_comparison"),
        "batch_size": config["cnn_batch_size"],
        "max_samples_image": config.get("sample_size", 3000),
        "max_samples_text": config.get("sample_size", 1000) // 2,
        "epochs_image": min(3, config.get("cnn_epochs", 3)),
        "epochs_text": min(2, config.get("glue_epochs", 2)),
        "comparison_samples": config.get("gradcam_samples", 16),
        "comparison_samples_text": 10
    }
    
    comparison = XAIMethodsComparison(comparison_config)
    
    # Determine which experiments to run:
    # - If neither --diet-images nor --diet-text specified, run both
    # - If only --diet-images specified, run images only
    # - If only --diet-text specified, run text only
    # - If both specified, run both
    run_images = args.diet_images or not args.diet_text
    run_text = args.diet_text or not args.diet_images
    
    # Use skip_training flag to reuse previously trained models
    results = comparison.run_full_comparison(
        run_images=run_images, 
        run_text=run_text,
        skip_training=args.skip_training
    )
    
    return results


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="XAI Experiments Orchestration Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_xai_experiments.py --all                    # Run all experiments
  python run_xai_experiments.py --cifar10 --epochs 5     # Run CIFAR-10 only
  python run_xai_experiments.py --glue                   # Run GLUE SST-2 only
  python run_xai_experiments.py --compare                # Run model comparison
  python run_xai_experiments.py --diet                   # Run DiET comparison (images + text)
  python run_xai_experiments.py --diet --diet-images     # DiET for images only
  python run_xai_experiments.py --diet --diet-text       # DiET for text only
  python run_xai_experiments.py --all --low-vram         # Low VRAM mode
  python run_xai_experiments.py --cifar10 --skip-training  # Skip training, use saved model
        """
    )
    
    # Experiment selection
    parser.add_argument("--all", action="store_true",
                        help="Run all experiments")
    parser.add_argument("--cifar10", action="store_true",
                        help="Run CIFAR-10 GradCAM experiment")
    parser.add_argument("--glue", action="store_true",
                        help="Run GLUE SST-2 BERT experiment")
    parser.add_argument("--compare", action="store_true",
                        help="Run model comparison experiment")
    parser.add_argument("--diet", action="store_true",
                        help="Run DiET vs basic XAI comparison")
    parser.add_argument("--diet-images", action="store_true",
                        help="Run DiET comparison for images only (with --diet)")
    parser.add_argument("--diet-text", action="store_true",
                        help="Run DiET comparison for text only (with --diet)")
    
    # Common options
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Data directory (default: ./data)")
    parser.add_argument("--output-dir", type=str, default="./outputs/xai_experiments",
                        help="Output directory (default: ./outputs/xai_experiments)")
    
    # Hardware options
    parser.add_argument("--low-vram", action="store_true",
                        help="Use low VRAM configuration for GPUs with < 8GB")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU mode")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device ID (default: 0)")
    
    # Training options
    parser.add_argument("--epochs", type=int,
                        help="Override number of training epochs")
    parser.add_argument("--batch-size", type=int,
                        help="Override batch size")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, use pretrained models")
    
    # Model options
    parser.add_argument("--model-type", type=str, default="resnet",
                        choices=["simple", "resnet"],
                        help="CNN model type (default: resnet)")
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # If no experiment selected, show help
    if not (args.all or args.cifar10 or args.glue or args.compare or args.diet):
        print("No experiment selected. Use --help for usage information.")
        print("Quick start: python run_xai_experiments.py --all")
        print("For DiET comparison: python run_xai_experiments.py --diet")
        return
    
    # Set GPU
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    # Get device configuration
    print("=" * 60)
    print("XAI Experiments Orchestration Script")
    print("=" * 60)
    
    config = get_device_config(args.low_vram)
    
    # Override config if command line args provided
    if args.epochs:
        config["cnn_epochs"] = args.epochs
        config["glue_epochs"] = args.epochs
    if args.batch_size:
        config["batch_size"] = args.batch_size
        config["cnn_batch_size"] = args.batch_size
        config["glue_batch_size"] = max(MIN_GLUE_BATCH_SIZE, args.batch_size // 4)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Store all results
    all_results = {
        "config": config,
        "experiments": {}
    }
    
    total_start = time.time()
    
    # Run selected experiments
    try:
        if args.all or args.cifar10:
            all_results["experiments"]["cifar10"] = run_cifar10_experiment(config, args)
        
        if args.all or args.glue:
            all_results["experiments"]["glue_sst2"] = run_glue_experiment(config, args)
        
        if args.all or args.compare:
            all_results["experiments"]["model_comparison"] = run_model_comparison(config, args)
        
        if args.all or args.diet:
            all_results["experiments"]["diet_comparison"] = run_diet_experiment(config, args)
        
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
    except Exception as e:
        print(f"\n\nError during experiment: {e}")
        import traceback
        traceback.print_exc()
    
    total_time = time.time() - total_start
    all_results["total_time_seconds"] = total_time
    
    # Save overall results
    results_path = os.path.join(args.output_dir, "all_results.json")
    with open(results_path, "w") as f:
        # Make JSON serializable
        def make_serializable(obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, (int, float, bool, str, type(None))):
                return obj
            return str(obj)
        
        json.dump(make_serializable(all_results), f, indent=2)
    
    # Final summary
    print("\n" + "=" * 60)
    print("XAI Experiments Complete!")
    print("=" * 60)
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"Results saved to: {args.output_dir}")
    
    if "cifar10" in all_results["experiments"]:
        print(f"\nCIFAR-10 Results:")
        cifar_results = all_results["experiments"]["cifar10"]
        if isinstance(cifar_results, dict) and "final_test_accuracy" in cifar_results:
            print(f"  Test Accuracy: {cifar_results['final_test_accuracy']:.2f}%")
    
    if "glue_sst2" in all_results["experiments"]:
        print(f"\nGLUE SST-2 Results:")
        glue_results = all_results["experiments"]["glue_sst2"]
        if isinstance(glue_results, dict) and "final_val_accuracy" in glue_results:
            print(f"  Validation Accuracy: {glue_results['final_val_accuracy']:.2f}%")
    
    if "diet_comparison" in all_results["experiments"]:
        print(f"\nDiET Comparison Results:")
        diet_results = all_results["experiments"]["diet_comparison"]
        if isinstance(diet_results, dict):
            if "image_experiments" in diet_results and diet_results["image_experiments"]:
                img = diet_results["image_experiments"]
                print(f"  Image (CIFAR-10):")
                print(f"    GradCAM score: {img.get('gradcam_mean_score', 'N/A')}")
                print(f"    DiET score: {img.get('diet_mean_score', 'N/A')}")
                if img.get("diet_better"):
                    print(f"    ✓ DiET improves attribution quality")
            if "text_experiments" in diet_results and diet_results["text_experiments"]:
                txt = diet_results["text_experiments"]
                print(f"  Text (SST-2):")
                print(f"    IG-DiET overlap: {txt.get('ig_diet_overlap', 'N/A')}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
