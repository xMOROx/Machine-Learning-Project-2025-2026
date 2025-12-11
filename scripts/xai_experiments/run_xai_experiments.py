#!/usr/bin/env python3
"""XAI Experiments Orchestration Script.

This script orchestrates all XAI experiments including:
- CIFAR-10 classification with GradCAM
- GLUE SST-2 with BERT and Integrated Gradients
- Model comparison (CNN, RF, LightGBM, SVM, Logistic Regression)

Optimized for RTX 3060 (12GB VRAM) with configurable memory usage.

Usage:
    python run_xai_experiments.py --all                    # Run all experiments
    python run_xai_experiments.py --cifar10                # Run CIFAR-10 only
    python run_xai_experiments.py --glue                   # Run GLUE SST-2 only
    python run_xai_experiments.py --compare                # Run model comparison
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
        if gpu_memory < 8 or low_vram:
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
            "glue_batch_size": 4,
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
    if not (args.all or args.cifar10 or args.glue or args.compare):
        print("No experiment selected. Use --help for usage information.")
        print("Quick start: python run_xai_experiments.py --all")
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
        config["glue_batch_size"] = max(4, args.batch_size // 4)
    
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
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
