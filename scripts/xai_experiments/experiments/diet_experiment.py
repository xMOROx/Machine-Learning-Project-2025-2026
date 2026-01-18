"""DiET (Distractor Erasure Tuning) Experiment for XAI Comparison.

This module provides DiET implementation adapted for:
1. CIFAR-10 image classification with discriminative feature attribution
2. Integration with existing GradCAM for comparison
3. Evaluation metrics: pixel perturbation, faithfulness, localization
4. Resumable training with checkpoint support

Reference: Bhalla et al., "Discriminative Feature Attributions:
Bridging Post Hoc Explainability and Inherent Interpretability", NeurIPS 2023
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from typing import Dict, Any, List
from tqdm import tqdm


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from ..models.cnn import SimpleCNN, ResNetCIFAR
    from ..explainers.gradcam import GradCAM
    from ..utils.checkpointing import CheckpointManager
except ImportError:
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.cnn import SimpleCNN, ResNetCIFAR
    from explainers.gradcam import GradCAM
    from utils.checkpointing import CheckpointManager


class CIFAR10DatasetWithPreds(Dataset):
    """CIFAR-10 Dataset with model predictions for DiET training."""

    def __init__(
        self, images: torch.Tensor, labels: torch.Tensor, predictions: torch.Tensor
    ):
        self.images = images
        self.labels = labels
        self.predictions = predictions

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return idx, self.images[idx], self.labels[idx], self.predictions[idx]


class DiETExplainer:
    """DiET-based explainer for discriminative feature attribution.

    DiET creates masks that identify discriminative features by:
    1. Learning masks that, when applied, maintain model predictions
    2. Encouraging sparsity in the masks (focusing on important regions)
    3. Ensuring faithfulness to the original model
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        upsample_factor: int = 4,
        image_size: int = 32,
    ):
        """Initialize DiET explainer.

        Args:
            model: The base model to explain
            device: Device to run computations on
            upsample_factor: Factor for mask upsampling
            image_size: Input image size (CIFAR-10 = 32)
        """
        self.model = model
        self.device = device
        self.upsample_factor = upsample_factor
        self.image_size = image_size
        self.mask_size = image_size // upsample_factor

        self.model.eval()
        self.upsample = nn.Upsample(
            scale_factor=upsample_factor, mode="bilinear", align_corners=False
        )

    def get_predictions(
        self, images: torch.Tensor, batch_size: int = 64
    ) -> torch.Tensor:
        """Get model predictions for all images.

        Args:
            images: Tensor of images (N, C, H, W)
            batch_size: Batch size for prediction

        Returns:
            Softmax predictions tensor (N, num_classes)
        """
        self.model.eval()
        num_samples = len(images)
        predictions = []

        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                batch = images[i : i + batch_size].to(self.device)
                preds = F.softmax(self.model(batch), dim=1).cpu()
                predictions.append(preds)

        return torch.cat(predictions, dim=0)

    def _get_background(self, batch_size: int) -> torch.Tensor:
        """Generate random background for masked regions.

        Args:
            batch_size: Number of samples

        Returns:
            Background tensor for masked images (batch_size, C, H, W)
        """

        means = torch.ones((batch_size, 3)) * torch.tensor([0.5, 0.5, 0.5])
        stds = torch.ones((batch_size, 3)) * torch.tensor([0.05, 0.05, 0.05])
        background = torch.normal(mean=means, std=stds)
        background = background.unsqueeze(2).unsqueeze(3).clamp(0, 1)
        return background.to(self.device)

    def train_mask(
        self,
        mask: torch.Tensor,
        data_loader: DataLoader,
        mask_optimizer: optim.Optimizer,
        sparsity_weight: float,
    ) -> Dict[str, float]:
        """Train the mask for one epoch.

        Args:
            mask: Learnable mask tensor
            data_loader: DataLoader with images, labels, predictions
            mask_optimizer: Optimizer for mask
            sparsity_weight: Weight for sparsity loss

        Returns:
            Dictionary of training metrics
        """
        mask.requires_grad_(True)
        self.model.eval()

        total_loss = 0
        total_t1 = 0
        total_t2 = 0
        total_sparsity = 0
        total_faithful_acc = 0
        total_masked_acc = 0
        num_samples = 0

        for idx, images, labels, pred_original in data_loader:
            images = images.to(self.device)
            pred_original = pred_original.to(self.device)
            labels = labels.to(self.device)

            batch_mask = self.upsample(mask[idx]).to(self.device)

            background = self._get_background(len(idx))

            pred_full = F.softmax(self.model(images), dim=1)

            masked_images = batch_mask * images + (1 - batch_mask) * background
            pred_masked = F.softmax(self.model(masked_images), dim=1)

            t1 = torch.norm(pred_original - pred_full, p=1, dim=1).sum()

            t2 = torch.norm(pred_full - pred_masked, p=1, dim=1).sum()

            sparsity = torch.norm(batch_mask.view(len(idx), -1), p=1, dim=1).sum()
            sparsity = sparsity / (self.image_size * self.image_size)

            loss = (sparsity_weight * sparsity + t1 + t2) / len(idx)

            mask_optimizer.zero_grad()
            loss.backward()
            mask_optimizer.step()

            with torch.no_grad():
                mask.clamp_(0, 1)

            with torch.no_grad():
                faithful_acc = (
                    (pred_original.argmax(1) == pred_masked.argmax(1)).float().sum()
                )
                masked_acc = (pred_masked.argmax(1) == labels).float().sum()

            total_loss += loss.item() * len(idx)
            total_t1 += t1.item()
            total_t2 += t2.item()
            total_sparsity += sparsity.item()
            total_faithful_acc += faithful_acc.item()
            total_masked_acc += masked_acc.item()
            num_samples += len(labels)

        return {
            "loss": total_loss / num_samples,
            "t1_faithfulness": total_t1 / num_samples,
            "t2_consistency": total_t2 / num_samples,
            "sparsity": total_sparsity / num_samples,
            "faithful_acc": total_faithful_acc / num_samples,
            "masked_acc": total_masked_acc / num_samples,
        }

    def train_model(
        self,
        mask: torch.Tensor,
        data_loader: DataLoader,
        model_optimizer: optim.Optimizer,
    ) -> Dict[str, float]:
        """Fine-tune the model with DiET objective.

        Args:
            mask: Fixed mask tensor (not trained here)
            data_loader: DataLoader with images, labels, predictions
            model_optimizer: Optimizer for model

        Returns:
            Dictionary of training metrics
        """
        mask.requires_grad_(False)
        self.model.train()

        total_loss = 0
        total_t1 = 0
        total_t2 = 0
        total_faithful_acc = 0
        total_masked_acc = 0
        num_samples = 0

        for idx, images, labels, pred_original in data_loader:
            images = images.to(self.device)
            pred_original = pred_original.to(self.device)
            labels = labels.to(self.device)

            batch_mask = self.upsample(mask[idx]).to(self.device)
            background = self._get_background(len(idx))

            pred_full = F.softmax(self.model(images), dim=1)
            masked_images = batch_mask * images + (1 - batch_mask) * background
            pred_masked = F.softmax(self.model(masked_images), dim=1)

            t1 = torch.norm(pred_original - pred_full, p=1, dim=1).sum()
            t2 = torch.norm(pred_full - pred_masked, p=1, dim=1).sum()
            loss = (t1 + t2) / len(idx)

            model_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()

            with torch.no_grad():
                faithful_acc = (
                    (pred_original.argmax(1) == pred_masked.argmax(1)).float().sum()
                )
                masked_acc = (pred_masked.argmax(1) == labels).float().sum()

            total_loss += loss.item() * len(idx)
            total_t1 += t1.item()
            total_t2 += t2.item()
            total_faithful_acc += faithful_acc.item()
            total_masked_acc += masked_acc.item()
            num_samples += len(labels)

        self.model.eval()

        return {
            "loss": total_loss / num_samples,
            "t1_faithfulness": total_t1 / num_samples,
            "t2_consistency": total_t2 / num_samples,
            "faithful_acc": total_faithful_acc / num_samples,
            "masked_acc": total_masked_acc / num_samples,
        }

    def generate_attribution(self, _: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
        """Generate attribution map from trained mask.

        Args:
            image: Single image tensor (1, C, H, W)
            mask: Corresponding mask tensor (1, 1, H', W')

        Returns:
            Attribution map as numpy array (H, W)
        """
        with torch.no_grad():
            upsampled_mask = self.upsample(mask.unsqueeze(0)).squeeze()
            return upsampled_mask.cpu().numpy()


class DiETExperiment:
    """DiET Experiment for CIFAR-10 with comparison to basic XAI methods.

    This experiment:
    1. Trains a baseline model on CIFAR-10
    2. Applies DiET to learn discriminative masks
    3. Compares DiET attributions with GradCAM
    4. Evaluates using pixel perturbation and faithfulness metrics
    """

    CIFAR10_CLASSES = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    def __init__(self, config: Dict[str, Any]):
        """Initialize DiET experiment.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.data_dir = config.get("data_dir", "./data/cifar10")
        self.output_dir = config.get("output_dir", "./outputs/xai_experiments/diet")
        os.makedirs(self.output_dir, exist_ok=True)

        model_type = config.get("model_type", "resnet")
        if model_type == "simple":
            self.model = SimpleCNN(num_classes=10).to(self.device)
        else:
            self.model = ResNetCIFAR(num_classes=10, pretrained=True).to(self.device)

        self.model_type = model_type

        self.upsample_factor = config.get("upsample_factor", 4)
        self.rounding_steps = config.get("rounding_steps", 3)
        self.mask_lr = config.get("mask_lr", 500)
        self.model_lr = config.get("model_lr", 0.0001)

        self.train_loader = None
        self.test_loader = None
        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None

        self.results = {}

        # Initialize checkpoint manager for resumable training
        checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
        self.experiment_name = config.get("experiment_name", f"diet_image_{model_type}")

    def prepare_data(self, max_samples: int = 5000) -> None:
        """Prepare CIFAR-10 data.

        Args:
            max_samples: Maximum samples for DiET training (memory constraint)
        """
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=False, download=True, transform=transform
        )

        train_images_list = []
        train_labels_list = []

        for i in range(min(max_samples, len(train_dataset))):
            img, label = train_dataset[i]
            train_images_list.append(img)
            train_labels_list.append(label)

        self.train_images = torch.stack(train_images_list)
        self.train_labels = torch.tensor(train_labels_list)

        test_images_list = []
        test_labels_list = []

        for i in range(min(max_samples // 5, len(test_dataset))):
            img, label = test_dataset[i]
            test_images_list.append(img)
            test_labels_list.append(label)

        self.test_images = torch.stack(test_images_list)
        self.test_labels = torch.tensor(test_labels_list)

        batch_size = self.config.get("batch_size", 64)
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )

        print(f"Training samples (DiET): {len(self.train_images)}")
        print(f"Test samples (DiET): {len(self.test_images)}")

    def train_baseline(
        self, epochs: int = 5, save_checkpoint_every: int = 1
    ) -> Dict[str, Any]:
        """Train baseline model on CIFAR-10 with checkpoint support.

        Training can be resumed from the last checkpoint if interrupted.

        Args:
            epochs: Number of training epochs
            save_checkpoint_every: Save checkpoint every N epochs (default: 1)

        Returns:
            Training history
        """
        checkpoint_name = f"{self.experiment_name}_baseline"
        start_epoch = 0

        print(f"\nTraining baseline {self.model_type} model...")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        history = {"train_loss": [], "train_acc": [], "test_acc": []}

        # Check for existing checkpoint
        if self.checkpoint_manager.has_checkpoint(checkpoint_name):
            print("Found checkpoint, resuming training...")
            checkpoint = self.checkpoint_manager.load_checkpoint(
                checkpoint_name, self.device
            )
            if checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                if "scheduler_state_dict" in checkpoint:
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                start_epoch = checkpoint["epoch"] + 1
                history = checkpoint.get("extra_state", {}).get("history", history)
                print(f"Resuming from epoch {start_epoch}")

        dropout = nn.Dropout(p=0.5)

        for epoch in range(start_epoch, epochs):
            self.model.train()
            train_loss = 0
            correct = 0
            total = 0

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                if self.model.training:
                    spatial_mask = dropout(torch.ones((inputs.shape[0], 1, 32, 32), device=self.device)).clamp(0, 1)
                    inputs = inputs * spatial_mask + (1 - spatial_mask) * 0.5

                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                pbar.set_postfix(
                    {
                        "loss": f"{train_loss / (pbar.n + 1):.4f}",
                        "acc": f"{100 * correct / total:.2f}%",
                    }
                )

            scheduler.step()

            test_acc = self._evaluate_model()

            history["train_loss"].append(train_loss / len(self.train_loader))
            history["train_acc"].append(100 * correct / total)
            history["test_acc"].append(test_acc)

            print(
                f"Epoch {epoch + 1}: Train Acc: {history['train_acc'][-1]:.2f}%, Test Acc: {test_acc:.2f}%"
            )

            # Save checkpoint
            if (epoch + 1) % save_checkpoint_every == 0 or epoch == epochs - 1:
                self.checkpoint_manager.save_checkpoint(
                    checkpoint_name,
                    epoch=epoch,
                    model_state_dict=self.model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    scheduler_state_dict=scheduler.state_dict(),
                    metrics={
                        "train_acc": history["train_acc"][-1],
                        "test_acc": test_acc,
                    },
                    extra_state={"history": history},
                )

        baseline_path = os.path.join(self.output_dir, f"baseline_{self.model_type}.pth")
        torch.save(self.model.state_dict(), baseline_path)
        print(f"Baseline model saved to {baseline_path}")

        # Clean up checkpoint after successful completion
        self.checkpoint_manager.delete_checkpoint(checkpoint_name)

        self.results["baseline"] = {
            "final_train_acc": history["train_acc"][-1],
            "final_test_acc": history["test_acc"][-1],
            "history": history,
        }

        return history

    def _evaluate_model(self) -> float:
        """Evaluate model on test set."""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return 100 * correct / total

    def load_baseline(self, model_path: str = None) -> bool:
        """Load a previously trained baseline model.

        Args:
            model_path: Path to the model checkpoint. If None, uses default path.

        Returns:
            True if model was loaded successfully, False otherwise
        """
        if model_path is None:
            model_path = os.path.join(
                self.output_dir, f"baseline_{self.model_type}.pth"
            )

        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            return False

        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            test_acc = self._evaluate_model()
            print(f"Loaded baseline model from {model_path}")
            print(f"Test accuracy: {test_acc:.2f}%")

            self.results["baseline"] = {
                "final_test_acc": test_acc,
                "loaded_from": model_path,
            }
            return True
        except Exception as e:
            print(f"Failed to load model from {model_path}: {e}")
            return False

    def run_diet(self) -> Dict[str, Any]:
        """Run DiET distillation process.

        Returns:
            DiET training results
        """
        print("\n" + "=" * 50)
        print("Running DiET Distillation")
        print("=" * 50)

        diet = DiETExplainer(
            self.model,
            device=self.device,
            upsample_factor=self.upsample_factor,
            image_size=32,
        )

        print("Getting baseline predictions...")
        train_preds = diet.get_predictions(self.train_images)
        test_preds = diet.get_predictions(self.test_images)

        diet_dataset = CIFAR10DatasetWithPreds(
            self.train_images, self.train_labels, train_preds
        )
        diet_loader = DataLoader(diet_dataset, batch_size=64, shuffle=True)

        test_diet_dataset = CIFAR10DatasetWithPreds(
            self.test_images, self.test_labels, test_preds
        )
        test_diet_loader = DataLoader(test_diet_dataset, batch_size=128, shuffle=False)

        mask_size = 32 // self.upsample_factor
        mask = torch.ones(
            (len(train_preds), 1, mask_size, mask_size), requires_grad=True
        )
        mask_optimizer = optim.SGD([mask], lr=self.mask_lr)
        model_optimizer = optim.Adam(self.model.parameters(), lr=self.model_lr)

        rounding_scheme = [
            0.4 - r * (0.4 / self.rounding_steps) for r in range(self.rounding_steps)
        ]
        sparsity_weights = [
            1 - r * (0.9 / self.rounding_steps) for r in range(self.rounding_steps)
        ]

        diet_history = []

        for step in range(self.rounding_steps):
            print(f"\n--- DiET Step {step + 1}/{self.rounding_steps} ---")

            print("Training mask...")
            prev_loss = float("inf")
            prev_prev_loss = float("inf")
            max_mask_iters = 50

            for i in range(max_mask_iters):
                metrics = diet.train_mask(mask, diet_loader, mask_optimizer, sparsity_weights[step])
                
                # Check convergence (within 0.5% tolerance)
                mask_converged = (
                    metrics["loss"] >= 0.995 * prev_prev_loss and 
                    metrics["loss"] <= 1.005 * prev_prev_loss
                )
                
                if i % 5 == 0:
                    print(f"  Iter {i}: Loss={metrics['loss']:.4f}, Sparsity={metrics['sparsity']:.4f}")
                
                if mask_converged and i > 10:  
                    print(f"  Mask converged at iteration {i}")
                    break
                
                prev_prev_loss = prev_loss
                prev_loss = metrics["loss"]

            with torch.no_grad():
                mask.copy_(torch.round(mask + rounding_scheme[step]).clamp(0, 1))

            mask_path = os.path.join(self.output_dir, f"diet_mask_step{step}.pt")
            torch.save(mask.detach(), mask_path)

            print("Training model...")
            prev_loss = float("inf")
            for i in range(30):
                metrics = diet.train_model(mask, diet_loader, model_optimizer)

                if metrics["loss"] < 0.025 or abs(metrics["loss"] - prev_loss) < 0.001:
                    break
                prev_loss = metrics["loss"]

                if i % 10 == 0:
                    print(
                        f"  Iter {i}: loss={metrics['loss']:.4f}, faithful_acc={metrics['faithful_acc']:.4f}"
                    )

            diet_history.append(
                {
                    "step": step,
                    "mask_metrics": metrics,
                    "model_test_acc": self._evaluate_model(),
                }
            )

            print(
                f"Step {step + 1} complete: Test Acc = {diet_history[-1]['model_test_acc']:.2f}%"
            )

        final_model_path = os.path.join(self.output_dir, f"diet_{self.model_type}.pth")
        torch.save(self.model.state_dict(), final_model_path)

        self.diet_mask = mask.detach()

        self.results["diet"] = {
            "history": diet_history,
            "final_test_acc": self._evaluate_model(),
        }

        return self.results["diet"]

    def generate_gradcam_attributions(self, num_samples: int = 16) -> List[np.ndarray]:
        """Generate GradCAM attributions for comparison.

        Args:
            num_samples: Number of samples

        Returns:
            List of GradCAM heatmaps
        """
        gradcam = GradCAM(self.model, self.device)
        heatmaps = []

        indices = np.random.choice(len(self.test_images), num_samples, replace=False)

        for idx in indices:
            img = self.test_images[idx : idx + 1]
            cam, _, _ = gradcam.generate_cam(img)
            heatmaps.append(cam)

        return heatmaps, indices

    def compare_methods(self, num_samples: int = 16) -> Dict[str, Any]:
        """Compare DiET with GradCAM."""
        print("\n" + "=" * 50)
        print("Comparing DiET vs GradCAM")
        print("=" * 50)

        gradcam_maps, indices = self.generate_gradcam_attributions(num_samples)

        diet_explainer = DiETExplainer(
            self.model, self.device, self.upsample_factor, 32
        )
        diet_maps = []

        for idx in indices:
            mask_slice = self.diet_mask[idx % len(self.diet_mask)]
            diet_map = diet_explainer.generate_attribution(
                self.test_images[idx : idx + 1], mask_slice
            )
            diet_maps.append(diet_map)

        # Basic pixel perturbation
        gradcam_scores = self._pixel_perturbation_test(gradcam_maps, indices)
        diet_scores = self._pixel_perturbation_test(diet_maps, indices)

        try:
            from ..metrics.attribution_metrics import (
                AOPC, FaithfulnessCorrelation, InsertionDeletion
            )
            
            test_images_subset = torch.stack([self.test_images[i] for i in indices])
            test_labels_subset = torch.tensor([self.test_labels[i] for i in indices])
            
            # AOPC
            aopc_metric = AOPC(self.model, self.device, num_steps=10, patch_size=4)
            gradcam_aopc_result = aopc_metric.compute(
                test_images_subset, test_labels_subset, gradcam_maps
            )
            diet_aopc_result = aopc_metric.compute(
                test_images_subset, test_labels_subset, diet_maps
            )
            
            # Faithfulness Correlation
            faith_metric = FaithfulnessCorrelation(self.model, self.device, num_samples=30)
            gradcam_faith_result = faith_metric.compute(
                test_images_subset, test_labels_subset, gradcam_maps
            )
            diet_faith_result = faith_metric.compute(
                test_images_subset, test_labels_subset, diet_maps
            )
            
            # Insertion/Deletion
            ins_del_metric = InsertionDeletion(self.model, self.device, num_steps=50)
            gradcam_ins_del_result = ins_del_metric.compute(
                test_images_subset, test_labels_subset, gradcam_maps
            )
            diet_ins_del_result = ins_del_metric.compute(
                test_images_subset, test_labels_subset, diet_maps
            )
            
            additional_metrics = {
                "gradcam_aopc": gradcam_aopc_result.value,
                "diet_aopc": diet_aopc_result.value,
                "gradcam_faithfulness": gradcam_faith_result.value,
                "diet_faithfulness": diet_faith_result.value,
                "gradcam_insertion_auc": gradcam_ins_del_result.details["insertion_auc"],
                "diet_insertion_auc": diet_ins_del_result.details["insertion_auc"],
                "gradcam_deletion_auc": gradcam_ins_del_result.details["deletion_auc"],
                "diet_deletion_auc": diet_ins_del_result.details["deletion_auc"],
            }
        except Exception as e:
            print(f"Warning: Could not compute advanced metrics: {e}")
            additional_metrics = {}

        comparison = {
            "gradcam": {
                "perturbation_scores": gradcam_scores,
                "mean_score": np.mean(list(gradcam_scores.values())),
            },
            "diet": {
                "perturbation_scores": diet_scores,
                "mean_score": np.mean(list(diet_scores.values())),
            },
            "improvement": np.mean(list(diet_scores.values()))
            - np.mean(list(gradcam_scores.values())),
            **additional_metrics,  # Add the new metrics here
        }

        self._save_comparison_visualizations(indices, gradcam_maps, diet_maps)

        self.results["comparison"] = comparison

        print(f"\nPixel Perturbation Results:")
        print(f"  GradCAM mean score: {comparison['gradcam']['mean_score']:.4f}")
        print(f"  DiET mean score: {comparison['diet']['mean_score']:.4f}")
        print(f"  Improvement: {comparison['improvement']:.4f}")
        
        if additional_metrics:
            print(f"\nAdvanced Metrics:")
            print(f"  AOPC - GradCAM: {additional_metrics.get('gradcam_aopc', 'N/A'):.4f}")
            print(f"  AOPC - DiET: {additional_metrics.get('diet_aopc', 'N/A'):.4f}")

        return comparison

    def _pixel_perturbation_test(
        self,
        attribution_maps: List[np.ndarray],
        indices: np.ndarray,
        percentages: List[int] = [10, 20, 50],
    ) -> Dict[int, float]:
        """Evaluate attributions using pixel perturbation.

        Keep top-k% pixels and measure accuracy.
        Higher accuracy means better attribution (important pixels identified).

        Args:
            attribution_maps: List of attribution maps
            indices: Sample indices
            percentages: Percentages of pixels to keep

        Returns:
            Dictionary of percentage -> accuracy
        """
        self.model.eval()
        results = {p: 0 for p in percentages}

        with torch.no_grad():
            for i, (attr_map, idx) in enumerate(zip(attribution_maps, indices)):
                image = self.test_images[idx : idx + 1].to(self.device)
                label = self.test_labels[idx].item()

                attr_tensor = torch.tensor(attr_map).float()
                if attr_tensor.dim() == 2:
                    attr_tensor = attr_tensor.unsqueeze(0).unsqueeze(0)
                attr_resized = F.interpolate(
                    attr_tensor, size=(32, 32), mode="bilinear", align_corners=False
                )
                attr_resized = attr_resized.squeeze().numpy()

                for p in percentages:
                    threshold = np.percentile(attr_resized, 100 - p)
                    mask = (attr_resized >= threshold).astype(np.float32)
                    mask = torch.tensor(mask).unsqueeze(0).unsqueeze(0).to(self.device)

                    masked_image = image * mask

                    pred = self.model(masked_image).argmax(1).item()
                    if pred == label:
                        results[p] += 1

        for p in percentages:
            results[p] = results[p] / len(indices)

        return results

    def _save_comparison_visualizations(
        self,
        indices: np.ndarray,
        gradcam_maps: List[np.ndarray],
        diet_maps: List[np.ndarray],
        max_vis: int = 8,
    ) -> None:
        """Save comparison visualizations."""
        viz_dir = os.path.join(self.output_dir, "comparison_visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        mean = torch.tensor([0.4914, 0.4822, 0.4465])
        std = torch.tensor([0.2023, 0.1994, 0.2010])

        num_vis = min(max_vis, len(indices))
        fig, axes = plt.subplots(num_vis, 4, figsize=(16, 4 * num_vis))

        for i in range(num_vis):
            idx = indices[i]
            img = self.test_images[idx].clone()

            for c in range(3):
                img[c] = img[c] * std[c] + mean[c]
            img = torch.clamp(img, 0, 1)
            img_np = img.permute(1, 2, 0).numpy()

            label = self.test_labels[idx].item()

            axes[i, 0].imshow(img_np)
            axes[i, 0].set_title(f"Original: {self.CIFAR10_CLASSES[label]}")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(img_np)
            gc_resized = np.array(
                torch.nn.functional.interpolate(
                    torch.tensor(gradcam_maps[i]).unsqueeze(0).unsqueeze(0),
                    size=(32, 32),
                    mode="bilinear",
                ).squeeze()
            )
            axes[i, 1].imshow(gc_resized, cmap="jet", alpha=0.5)
            axes[i, 1].set_title("GradCAM")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(img_np)
            diet_resized = np.array(
                torch.nn.functional.interpolate(
                    torch.tensor(diet_maps[i]).unsqueeze(0).unsqueeze(0),
                    size=(32, 32),
                    mode="bilinear",
                ).squeeze()
            )
            axes[i, 2].imshow(diet_resized, cmap="jet", alpha=0.5)
            axes[i, 2].set_title("DiET")
            axes[i, 2].axis("off")

            diff = diet_resized - gc_resized
            axes[i, 3].imshow(diff, cmap="RdBu", vmin=-1, vmax=1)
            axes[i, 3].set_title("DiET - GradCAM")
            axes[i, 3].axis("off")

        plt.tight_layout()
        plt.savefig(
            os.path.join(viz_dir, "diet_vs_gradcam.png"), dpi=150, bbox_inches="tight"
        )
        plt.close()

        print(f"Comparison visualizations saved to {viz_dir}")

    def run_full_experiment(self, skip_training: bool = False) -> Dict[str, Any]:
        """Run complete DiET experiment with comparison.

        Args:
            skip_training: If True, try to load previously trained baseline model
                          instead of training from scratch

        Returns:
            All experiment results
        """
        print("=" * 60)
        print("DiET Experiment: Discriminative Feature Attribution")
        print("=" * 60)

        start_time = time.time()

        print("\n[Step 1/4] Preparing data...")
        max_samples = self.config.get("max_samples", 5000)
        self.prepare_data(max_samples)

        if skip_training:
            print("\n[Step 2/4] Loading baseline model...")
            if not self.load_baseline():
                print("Failed to load saved model. Training from scratch...")
                baseline_epochs = self.config.get("baseline_epochs", 5)
                self.train_baseline(baseline_epochs)
        else:
            print("\n[Step 2/4] Training baseline model...")
            baseline_epochs = self.config.get("baseline_epochs", 5)
            self.train_baseline(baseline_epochs)

        print("\n[Step 3/4] Running DiET distillation...")
        self.run_diet()

        print("\n[Step 4/4] Comparing DiET vs GradCAM...")
        comparison_samples = self.config.get("comparison_samples", 16)
        self.compare_methods(comparison_samples)

        total_time = time.time() - start_time
        self.results["total_time_seconds"] = total_time

        results_path = os.path.join(self.output_dir, "diet_experiment_results.json")

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

        with open(results_path, "w") as f:
            json.dump(make_serializable(self.results), f, indent=2)

        print("\n" + "=" * 60)
        print("DiET Experiment Summary")
        print("=" * 60)
        print(
            f"Baseline Test Accuracy: {self.results['baseline']['final_test_acc']:.2f}%"
        )
        print(f"DiET Test Accuracy: {self.results['diet']['final_test_acc']:.2f}%")
        print("\nPixel Perturbation Comparison:")
        print(f"  GradCAM: {self.results['comparison']['gradcam']['mean_score']:.4f}")
        print(f"  DiET: {self.results['comparison']['diet']['mean_score']:.4f}")

        improvement = self.results["comparison"]["improvement"]
        if improvement > 0:
            print(
                f"\n✓ DiET shows {improvement:.4f} improvement in attribution quality!"
            )
        else:
            print(f"\n→ GradCAM performs {-improvement:.4f} better on this dataset")

        print(f"\nTotal time: {total_time:.1f} seconds")
        print(f"Results saved to {results_path}")
        print("=" * 60)

        return self.results
