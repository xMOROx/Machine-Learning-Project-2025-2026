"""CIFAR-10 Experiment with GradCAM Visualization.

This module provides a complete pipeline for:
1. Training CNN models on CIFAR-10
2. Evaluating model performance
3. Generating GradCAM explanations
4. Visualizing and saving results
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
from typing import Optional, Dict, Any, Tuple
from tqdm import tqdm

# Use relative imports when running as part of the package
try:
    from ..models.cnn import SimpleCNN, ResNetCIFAR
    from ..explainers.gradcam import GradCAM
except ImportError:
    # Fallback for direct script execution
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.cnn import SimpleCNN, ResNetCIFAR
    from explainers.gradcam import GradCAM


class CIFAR10Experiment:
    """CIFAR-10 Classification with GradCAM Explanation Pipeline.
    
    This class provides methods for:
    - Loading and preprocessing CIFAR-10 data
    - Training CNN models (SimpleCNN or ResNet)
    - Evaluating model performance
    - Generating GradCAM visualizations
    
    Attributes:
        config: Configuration dictionary with experiment parameters
        device: Device to run experiments on
        model: The CNN model
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
    """
    
    CIFAR10_CLASSES = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize CIFAR-10 experiment.
        
        Args:
            config: Configuration dictionary with:
                - data_dir: Path to store/load CIFAR-10 data
                - output_dir: Path to save results
                - model_type: "simple" or "resnet"
                - batch_size: Training batch size
                - num_workers: DataLoader workers
                - epochs: Training epochs
                - learning_rate: Initial learning rate
                - device: Device to use
        """
        self.config = config
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        # Set up directories
        self.data_dir = config.get("data_dir", "./data/cifar10")
        self.output_dir = config.get("output_dir", "./outputs/xai_experiments/cifar10")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize model
        model_type = config.get("model_type", "resnet")
        if model_type == "simple":
            self.model = SimpleCNN(num_classes=10).to(self.device)
        else:
            self.model = ResNetCIFAR(num_classes=10, pretrained=True).to(self.device)
        
        self.model_type = model_type
        
        # Data loaders
        self.train_loader = None
        self.test_loader = None
        
        # Training history
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": []
        }
    
    def prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        """Prepare CIFAR-10 data loaders.
        
        Returns:
            Tuple of (train_loader, test_loader)
        """
        # Define transforms
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        # Load datasets
        train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=transform_train
        )
        
        test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=transform_test
        )
        
        batch_size = self.config.get("batch_size", 64)
        num_workers = self.config.get("num_workers", 4)
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
        return self.train_loader, self.test_loader
    
    def train(self) -> Dict[str, list]:
        """Train the model.
        
        Returns:
            Training history dictionary
        """
        if self.train_loader is None:
            self.prepare_data()
        
        epochs = self.config.get("epochs", 10)
        lr = self.config.get("learning_rate", 0.001)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        print(f"\nTraining {self.model_type} model for {epochs} epochs")
        print(f"Device: {self.device}")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    "loss": f"{train_loss / (pbar.n + 1):.4f}",
                    "acc": f"{100. * train_correct / train_total:.2f}%"
                })
            
            # Evaluation phase
            test_loss, test_acc = self.evaluate()
            
            # Record history
            self.history["train_loss"].append(train_loss / len(self.train_loader))
            self.history["train_acc"].append(100. * train_correct / train_total)
            self.history["test_loss"].append(test_loss)
            self.history["test_acc"].append(test_acc)
            
            print(f"Epoch {epoch+1}: Train Acc: {self.history['train_acc'][-1]:.2f}%, "
                  f"Test Acc: {test_acc:.2f}%")
            
            scheduler.step()
        
        # Save model
        model_path = os.path.join(self.output_dir, f"{self.model_type}_cifar10.pth")
        torch.save(self.model.state_dict(), model_path)
        print(f"\nModel saved to {model_path}")
        
        # Save history
        history_path = os.path.join(self.output_dir, f"{self.model_type}_history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        
        return self.history
    
    def evaluate(self) -> Tuple[float, float]:
        """Evaluate model on test set.
        
        Returns:
            Tuple of (test_loss, test_accuracy)
        """
        if self.test_loader is None:
            self.prepare_data()
        
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return test_loss / len(self.test_loader), 100. * correct / total
    
    def load_model(self, model_path: str) -> None:
        """Load model from checkpoint.
        
        Args:
            model_path: Path to model checkpoint
        """
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded from {model_path}")
    
    def generate_gradcam(
        self,
        num_samples: int = 16,
        save_visualizations: bool = True
    ) -> Dict[str, Any]:
        """Generate GradCAM explanations for test samples.
        
        Args:
            num_samples: Number of samples to visualize
            save_visualizations: Whether to save visualizations
            
        Returns:
            Dictionary with GradCAM results
        """
        if self.test_loader is None:
            self.prepare_data()
        
        # Initialize GradCAM
        gradcam = GradCAM(self.model, self.device)
        
        # Get some test samples
        test_dataset = self.test_loader.dataset
        
        # Sample indices
        indices = np.random.choice(len(test_dataset), num_samples, replace=False)
        
        # Collect results
        images = []
        heatmaps = []
        predictions = []
        confidences = []
        true_labels = []
        
        # Get unnormalized transform for visualization
        mean = torch.tensor([0.4914, 0.4822, 0.4465])
        std = torch.tensor([0.2023, 0.1994, 0.2010])
        
        print(f"\nGenerating GradCAM for {num_samples} samples...")
        
        for idx in tqdm(indices):
            img, label = test_dataset[idx]
            
            # Generate CAM
            cam, pred_class, conf = gradcam.generate_cam(img.unsqueeze(0))
            
            # Unnormalize image for visualization
            img_unnorm = img.clone()
            for c in range(3):
                img_unnorm[c] = img_unnorm[c] * std[c] + mean[c]
            img_unnorm = torch.clamp(img_unnorm, 0, 1)
            
            # Convert to numpy (H, W, C)
            img_np = img_unnorm.permute(1, 2, 0).numpy()
            img_np = (img_np * 255).astype(np.uint8)
            
            images.append(img_np)
            heatmaps.append(cam)
            predictions.append(pred_class)
            confidences.append(conf)
            true_labels.append(label)
        
        # Visualize and save
        if save_visualizations:
            # Individual visualizations
            viz_dir = os.path.join(self.output_dir, "gradcam_visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            
            for i, (img, cam, pred, conf, true_lbl) in enumerate(
                zip(images, heatmaps, predictions, confidences, true_labels)
            ):
                save_path = os.path.join(viz_dir, f"sample_{i}.png")
                GradCAM.visualize(
                    img, cam, pred, conf,
                    class_names=self.CIFAR10_CLASSES,
                    save_path=save_path,
                    show=False
                )
            
            # Batch visualization
            batch_save_path = os.path.join(self.output_dir, "gradcam_batch.png")
            GradCAM.visualize_batch(
                np.array(images),
                np.array(heatmaps),
                np.array(predictions),
                np.array(confidences),
                class_names=self.CIFAR10_CLASSES,
                save_path=batch_save_path,
                show=False
            )
            
            print(f"Visualizations saved to {viz_dir}")
            print(f"Batch visualization saved to {batch_save_path}")
        
        # Calculate accuracy on GradCAM samples
        correct = sum(p == t for p, t in zip(predictions, true_labels))
        accuracy = 100 * correct / len(predictions)
        
        results = {
            "num_samples": num_samples,
            "accuracy_on_samples": accuracy,
            "predictions": predictions,
            "true_labels": true_labels,
            "confidences": confidences,
            "mean_confidence": np.mean(confidences)
        }
        
        # Save results
        results_path = os.path.join(self.output_dir, "gradcam_results.json")
        with open(results_path, "w") as f:
            # Convert numpy types to Python types for JSON
            json_results = {
                k: (v.tolist() if isinstance(v, np.ndarray) else 
                    float(v) if isinstance(v, (np.float32, np.float64)) else v)
                for k, v in results.items()
            }
            json.dump(json_results, f, indent=2)
        
        print(f"\nGradCAM Results:")
        print(f"  Accuracy on samples: {accuracy:.2f}%")
        print(f"  Mean confidence: {np.mean(confidences):.4f}")
        
        return results
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete CIFAR-10 experiment pipeline.
        
        Returns:
            Dictionary with all results
        """
        print("=" * 60)
        print("CIFAR-10 GradCAM Experiment Pipeline")
        print("=" * 60)
        
        start_time = time.time()
        
        # Step 1: Prepare data
        print("\n[Step 1/4] Preparing data...")
        self.prepare_data()
        
        # Step 2: Train model
        print("\n[Step 2/4] Training model...")
        self.train()
        
        # Step 3: Evaluate
        print("\n[Step 3/4] Final evaluation...")
        test_loss, test_acc = self.evaluate()
        print(f"Final Test Accuracy: {test_acc:.2f}%")
        
        # Step 4: Generate GradCAM
        print("\n[Step 4/4] Generating GradCAM explanations...")
        gradcam_results = self.generate_gradcam(num_samples=16)
        
        total_time = time.time() - start_time
        
        # Compile final results
        results = {
            "model_type": self.model_type,
            "config": self.config,
            "training_history": self.history,
            "final_test_accuracy": test_acc,
            "gradcam_results": gradcam_results,
            "total_time_seconds": total_time
        }
        
        # Save final results
        final_results_path = os.path.join(self.output_dir, "experiment_results.json")
        with open(final_results_path, "w") as f:
            # Make JSON serializable
            def make_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {k: make_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_serializable(item) for item in obj]
                return obj
            
            json.dump(make_serializable(results), f, indent=2)
        
        print("\n" + "=" * 60)
        print(f"Pipeline completed in {total_time:.1f} seconds")
        print(f"Results saved to {self.output_dir}")
        print("=" * 60)
        
        return results
