"""Model Comparison Experiment.

This module provides comparison of multiple ML models on classification tasks:
- Neural Networks (CNN)
- Random Forest
- LightGBM
- SVM
- Logistic Regression
"""

import os
import time
import json
import torch
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms


class ModelComparison:
    """Compare multiple ML models on CIFAR-10 classification.
    
    Models tested:
    - CNN (SimpleCNN and ResNet)
    - Random Forest
    - LightGBM (if available)
    - SVM
    - Logistic Regression
    
    Attributes:
        config: Configuration dictionary
        device: Device to run experiments on
        results: Dictionary storing comparison results
    """
    
    CIFAR10_CLASSES = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize model comparison experiment.
        
        Args:
            config: Configuration dictionary with:
                - data_dir: Path to store/load data
                - output_dir: Path to save results
                - sample_size: Number of samples for sklearn models
                - device: Device to use
        """
        self.config = config
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        self.data_dir = config.get("data_dir", "./data/cifar10")
        self.output_dir = config.get("output_dir", "./outputs/xai_experiments/model_comparison")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.results = {}
        
        # Data containers
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.train_loader = None
        self.test_loader = None
    
    def prepare_data(self, sample_size: Optional[int] = None) -> None:
        """Prepare CIFAR-10 data.
        
        For sklearn models, we flatten the images and use a subset.
        For CNN models, we keep the full dataset.
        
        Args:
            sample_size: Maximum samples for sklearn models (None = use all)
        """
        print("Loading CIFAR-10 data...")
        
        # Transform for CNN
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        # Load full datasets
        train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=False, download=True, transform=transform
        )
        
        # DataLoaders for CNN
        batch_size = self.config.get("batch_size", 64)
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )
        
        # Flatten data for sklearn models
        sample_size = sample_size or self.config.get("sample_size", 10000)
        
        # Extract data
        X_train_list = []
        y_train_list = []
        
        for i, (img, label) in enumerate(train_dataset):
            if i >= sample_size:
                break
            X_train_list.append(img.numpy().flatten())
            y_train_list.append(label)
        
        X_test_list = []
        y_test_list = []
        
        test_sample_size = min(sample_size // 5, len(test_dataset))
        for i, (img, label) in enumerate(test_dataset):
            if i >= test_sample_size:
                break
            X_test_list.append(img.numpy().flatten())
            y_test_list.append(label)
        
        self.X_train = np.array(X_train_list)
        self.y_train = np.array(y_train_list)
        self.X_test = np.array(X_test_list)
        self.y_test = np.array(y_test_list)
        
        # Standardize
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        
        print(f"Training samples (sklearn): {len(self.X_train)}")
        print(f"Test samples (sklearn): {len(self.X_test)}")
        print(f"Training samples (CNN): {len(train_dataset)}")
        print(f"Test samples (CNN): {len(test_dataset)}")
    
    def train_random_forest(self) -> Dict[str, Any]:
        """Train and evaluate Random Forest classifier."""
        print("\n[Random Forest] Training...")
        start_time = time.time()
        
        n_estimators = self.config.get("rf_n_estimators", 100)
        max_depth = self.config.get("rf_max_depth", 20)
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=42
        )
        
        model.fit(self.X_train, self.y_train)
        
        # Evaluate
        train_pred = model.predict(self.X_train)
        test_pred = model.predict(self.X_test)
        
        train_acc = accuracy_score(self.y_train, train_pred) * 100
        test_acc = accuracy_score(self.y_test, test_pred) * 100
        
        training_time = time.time() - start_time
        
        result = {
            "model": "Random Forest",
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "training_time": training_time,
            "n_estimators": n_estimators,
            "max_depth": max_depth
        }
        
        print(f"[Random Forest] Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, Time: {training_time:.1f}s")
        
        self.results["random_forest"] = result
        return result
    
    def train_lightgbm(self) -> Dict[str, Any]:
        """Train and evaluate LightGBM classifier."""
        print("\n[LightGBM] Training...")
        
        try:
            import lightgbm as lgb
        except ImportError:
            print("[LightGBM] Not installed, skipping...")
            return {"model": "LightGBM", "error": "Not installed"}
        
        start_time = time.time()
        
        n_estimators = self.config.get("lgb_n_estimators", 100)
        learning_rate = self.config.get("lgb_learning_rate", 0.1)
        
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=31,
            n_jobs=-1,
            random_state=42,
            verbose=-1
        )
        
        model.fit(self.X_train, self.y_train)
        
        # Evaluate
        train_pred = model.predict(self.X_train)
        test_pred = model.predict(self.X_test)
        
        train_acc = accuracy_score(self.y_train, train_pred) * 100
        test_acc = accuracy_score(self.y_test, test_pred) * 100
        
        training_time = time.time() - start_time
        
        result = {
            "model": "LightGBM",
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "training_time": training_time,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate
        }
        
        print(f"[LightGBM] Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, Time: {training_time:.1f}s")
        
        self.results["lightgbm"] = result
        return result
    
    def train_svm(self) -> Dict[str, Any]:
        """Train and evaluate SVM classifier."""
        print("\n[SVM] Training...")
        start_time = time.time()
        
        # Use smaller subset for SVM (slow)
        svm_sample_size = min(5000, len(self.X_train))
        X_train_svm = self.X_train[:svm_sample_size]
        y_train_svm = self.y_train[:svm_sample_size]
        
        kernel = self.config.get("svm_kernel", "rbf")
        C = self.config.get("svm_C", 1.0)
        
        model = SVC(
            kernel=kernel,
            C=C,
            random_state=42,
            cache_size=2000
        )
        
        model.fit(X_train_svm, y_train_svm)
        
        # Evaluate
        train_pred = model.predict(X_train_svm)
        test_pred = model.predict(self.X_test)
        
        train_acc = accuracy_score(y_train_svm, train_pred) * 100
        test_acc = accuracy_score(self.y_test, test_pred) * 100
        
        training_time = time.time() - start_time
        
        result = {
            "model": "SVM",
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "training_time": training_time,
            "kernel": kernel,
            "C": C,
            "train_samples_used": svm_sample_size
        }
        
        print(f"[SVM] Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, Time: {training_time:.1f}s")
        
        self.results["svm"] = result
        return result
    
    def train_logistic_regression(self) -> Dict[str, Any]:
        """Train and evaluate Logistic Regression classifier."""
        print("\n[Logistic Regression] Training...")
        start_time = time.time()
        
        max_iter = self.config.get("lr_max_iter", 1000)
        
        model = LogisticRegression(
            max_iter=max_iter,
            n_jobs=-1,
            random_state=42,
            solver='saga',
            multi_class='multinomial'
        )
        
        model.fit(self.X_train, self.y_train)
        
        # Evaluate
        train_pred = model.predict(self.X_train)
        test_pred = model.predict(self.X_test)
        
        train_acc = accuracy_score(self.y_train, train_pred) * 100
        test_acc = accuracy_score(self.y_test, test_pred) * 100
        
        training_time = time.time() - start_time
        
        result = {
            "model": "Logistic Regression",
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "training_time": training_time,
            "max_iter": max_iter
        }
        
        print(f"[Logistic Regression] Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, Time: {training_time:.1f}s")
        
        self.results["logistic_regression"] = result
        return result
    
    def train_cnn(self, model_type: str = "simple") -> Dict[str, Any]:
        """Train and evaluate CNN classifier.
        
        Args:
            model_type: "simple" or "resnet"
        """
        try:
            from ..models.cnn import SimpleCNN, ResNetCIFAR
        except ImportError:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from models.cnn import SimpleCNN, ResNetCIFAR
        
        print(f"\n[CNN-{model_type}] Training...")
        start_time = time.time()
        
        # Initialize model
        if model_type == "simple":
            model = SimpleCNN(num_classes=10).to(self.device)
        else:
            model = ResNetCIFAR(num_classes=10, pretrained=True).to(self.device)
        
        epochs = self.config.get("cnn_epochs", 5)
        lr = self.config.get("cnn_learning_rate", 0.001)
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Training
        model.train()
        for epoch in range(epochs):
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        # Evaluation
        model.eval()
        train_correct = 0
        train_total = 0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            # Train accuracy (sample)
            for inputs, labels in list(self.train_loader)[:50]:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            # Test accuracy
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        test_acc = 100 * test_correct / test_total
        
        training_time = time.time() - start_time
        
        result = {
            "model": f"CNN-{model_type}",
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "training_time": training_time,
            "epochs": epochs,
            "learning_rate": lr
        }
        
        print(f"[CNN-{model_type}] Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, Time: {training_time:.1f}s")
        
        self.results[f"cnn_{model_type}"] = result
        return result
    
    def run_comparison(self) -> Dict[str, Any]:
        """Run all model comparisons.
        
        Returns:
            Dictionary with all comparison results
        """
        print("=" * 60)
        print("Model Comparison on CIFAR-10")
        print("=" * 60)
        
        total_start = time.time()
        
        # Prepare data
        print("\n[Step 1] Preparing data...")
        self.prepare_data()
        
        # Train all models
        print("\n[Step 2] Training models...")
        
        # Traditional ML models
        self.train_logistic_regression()
        self.train_random_forest()
        self.train_lightgbm()
        self.train_svm()
        
        # Deep learning models
        self.train_cnn("simple")
        self.train_cnn("resnet")
        
        total_time = time.time() - total_start
        
        # Create summary
        summary = {
            "comparison_results": self.results,
            "total_time_seconds": total_time,
            "config": {k: str(v) if not isinstance(v, (int, float, bool, str, list, dict, type(None))) else v 
                       for k, v in self.config.items()}
        }
        
        # Sort by test accuracy
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1].get("test_accuracy", 0),
            reverse=True
        )
        
        print("\n" + "=" * 60)
        print("Model Comparison Summary (sorted by test accuracy)")
        print("=" * 60)
        print(f"{'Model':<25} {'Train Acc':<12} {'Test Acc':<12} {'Time (s)':<10}")
        print("-" * 60)
        
        for name, result in sorted_results:
            if "error" not in result:
                print(f"{result['model']:<25} {result['train_accuracy']:<12.2f} {result['test_accuracy']:<12.2f} {result['training_time']:<10.1f}")
        
        print("=" * 60)
        print(f"Total time: {total_time:.1f}s")
        
        # Save results
        results_path = os.path.join(self.output_dir, "comparison_results.json")
        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results saved to {results_path}")
        
        return summary
