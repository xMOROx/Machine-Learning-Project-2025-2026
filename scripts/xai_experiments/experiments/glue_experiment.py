"""GLUE SST-2 Experiment with BERT and Integrated Gradients.

This module provides a complete pipeline for:
1. Fine-tuning BERT on GLUE SST-2 (sentiment classification)
2. Evaluating model performance
3. Generating Integrated Gradients explanations
4. Visualizing token attributions
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, Any, Tuple, List
from tqdm import tqdm


class SST2Dataset(Dataset):
    """Dataset class for SST-2 data."""

    def __init__(
        self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long),
            "text": text,
        }


class GLUEExperiment:
    """GLUE SST-2 Classification with Integrated Gradients Explanation Pipeline.

    This class provides methods for:
    - Loading and preprocessing GLUE SST-2 data
    - Fine-tuning BERT model
    - Evaluating model performance
    - Generating Integrated Gradients visualizations

    Attributes:
        config: Configuration dictionary with experiment parameters
        device: Device to run experiments on
        model: The BERT model wrapper
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
    """

    SST2_LABELS = ["negative", "positive"]

    def __init__(self, config: Dict[str, Any]):
        """Initialize GLUE SST-2 experiment.

        Args:
            config: Configuration dictionary with:
                - data_dir: Path to store/load GLUE data
                - output_dir: Path to save results
                - model_name: BERT model name (e.g., "bert-base-uncased")
                - batch_size: Training batch size
                - max_length: Maximum sequence length
                - epochs: Training epochs
                - learning_rate: Initial learning rate
                - device: Device to use
        """
        self.config = config
        self.device = config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.data_dir = config.get("data_dir", "./data/glue_sst2")
        self.output_dir = config.get(
            "output_dir", "./outputs/xai_experiments/glue_sst2"
        )
        os.makedirs(self.output_dir, exist_ok=True)

        self.model = None
        self.model_name = config.get("model_name", "bert-base-uncased")

        self.train_loader = None
        self.val_loader = None
        self.test_texts = None
        self.test_labels = None

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

    def _init_model(self):
        """Initialize the BERT model."""
        if self.model is None:
            try:
                from ..models.transformer import BertForSequenceClassificationWithIG
            except ImportError:
                import sys

                sys.path.insert(
                    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )
                from models.transformer import BertForSequenceClassificationWithIG
            self.model = BertForSequenceClassificationWithIG(
                model_name=self.model_name, num_labels=2, device=self.device
            )

    def prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        """Prepare SST-2 data loaders.

        Returns:
            Tuple of (train_loader, val_loader)
        """
        print("Loading SST-2 dataset...")

        from datasets import load_dataset

        dataset = load_dataset("glue", "sst2", cache_dir=self.data_dir)

        self._init_model()

        train_texts = dataset["train"]["sentence"]
        train_labels = dataset["train"]["label"]

        val_texts = dataset["validation"]["sentence"]
        val_labels = dataset["validation"]["label"]

        self.test_texts = val_texts[:100]
        self.test_labels = val_labels[:100]

        train_dataset = SST2Dataset(
            train_texts,
            train_labels,
            self.model.tokenizer,
            max_length=self.config.get("max_length", 128),
        )

        val_dataset = SST2Dataset(
            val_texts,
            val_labels,
            self.model.tokenizer,
            max_length=self.config.get("max_length", 128),
        )

        batch_size = self.config.get("batch_size", 16)

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )

        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")

        return self.train_loader, self.val_loader

    def train(self) -> Dict[str, list]:
        """Train the BERT model.

        Returns:
            Training history dictionary
        """
        if self.train_loader is None:
            self.prepare_data()

        epochs = self.config.get("epochs", 3)
        lr = self.config.get("learning_rate", 2e-5)

        self.model.train_mode()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.get_parameters(), lr=lr)

        total_steps = len(self.train_loader) * epochs
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps
        )

        print(f"\nFine-tuning BERT on SST-2 for {epochs} epochs")
        print(f"Device: {self.device}")

        for epoch in range(epochs):
            self.model.train_mode()
            train_loss = 0
            train_correct = 0
            train_total = 0

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            for batch in pbar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()

                pbar.set_postfix(
                    {
                        "loss": f"{train_loss / (pbar.n + 1):.4f}",
                        "acc": f"{100.0 * train_correct / train_total:.2f}%",
                    }
                )

            val_loss, val_acc = self.evaluate()

            self.history["train_loss"].append(train_loss / len(self.train_loader))
            self.history["train_acc"].append(100.0 * train_correct / train_total)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            print(
                f"Epoch {epoch + 1}: Train Acc: {self.history['train_acc'][-1]:.2f}%, "
                f"Val Acc: {val_acc:.2f}%"
            )

        model_path = os.path.join(self.output_dir, "bert_sst2")
        self.model.save_model(model_path)
        print(f"\nModel saved to {model_path}")

        history_path = os.path.join(self.output_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        return self.history

    def evaluate(self) -> Tuple[float, float]:
        """Evaluate model on validation set.

        Returns:
            Tuple of (val_loss, val_accuracy)
        """
        if self.val_loader is None:
            self.prepare_data()

        self.model.eval_mode()
        criterion = nn.CrossEntropyLoss()

        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return val_loss / len(self.val_loader), 100.0 * correct / total

    def load_model(self, model_path: str) -> None:
        """Load model from checkpoint.

        Args:
            model_path: Path to model checkpoint
        """
        self._init_model()
        self.model.load_model(model_path)
        self.model.eval_mode()
        print(f"Model loaded from {model_path}")

    def generate_integrated_gradients(
        self, num_samples: int = 10, n_steps: int = 50, save_visualizations: bool = True
    ) -> Dict[str, Any]:
        """Generate Integrated Gradients explanations for test samples.

        Args:
            num_samples: Number of samples to visualize
            n_steps: Number of integration steps
            save_visualizations: Whether to save visualizations

        Returns:
            Dictionary with IG results
        """
        if self.test_texts is None:
            self.prepare_data()

        try:
            from ..explainers.integrated_gradients import IntegratedGradientsText
        except ImportError:
            import sys

            sys.path.insert(
                0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            from explainers.integrated_gradients import IntegratedGradientsText

        ig = IntegratedGradientsText(self.model, self.device)

        sample_indices = np.random.choice(
            len(self.test_texts), min(num_samples, len(self.test_texts)), replace=False
        )

        results = []

        print(f"\nGenerating Integrated Gradients for {num_samples} samples...")

        viz_dir = os.path.join(self.output_dir, "ig_visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        for i, idx in enumerate(tqdm(sample_indices)):
            text = self.test_texts[idx]
            true_label = self.test_labels[idx]

            try:
                attributions, tokens, pred_class, confidence = ig.compute_attributions(
                    text, n_steps=n_steps
                )

                sample_result = {
                    "text": text,
                    "true_label": true_label,
                    "predicted_label": pred_class,
                    "confidence": confidence,
                    "correct": pred_class == true_label,
                    "tokens": tokens[:50],
                    "top_attributions": [],
                }

                valid_indices = [
                    j
                    for j, t in enumerate(tokens)
                    if t not in ["[PAD]", "[CLS]", "[SEP]"]
                ]

                if valid_indices:
                    valid_attrs = [(tokens[j], attributions[j]) for j in valid_indices]
                    sorted_attrs = sorted(
                        valid_attrs, key=lambda x: abs(x[1]), reverse=True
                    )
                    sample_result["top_attributions"] = [
                        {"token": t, "attribution": float(a)}
                        for t, a in sorted_attrs[:10]
                    ]

                results.append(sample_result)

                if save_visualizations:
                    save_path = os.path.join(viz_dir, f"sample_{i}.png")
                    IntegratedGradientsText.visualize(
                        tokens,
                        attributions,
                        pred_class,
                        confidence,
                        class_names=self.SST2_LABELS,
                        save_path=save_path,
                        show=False,
                    )

                    html_path = os.path.join(viz_dir, f"sample_{i}.html")
                    IntegratedGradientsText.visualize_html(
                        tokens,
                        attributions,
                        pred_class,
                        confidence,
                        class_names=self.SST2_LABELS,
                        save_path=html_path,
                    )

            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue

        correct = sum(r["correct"] for r in results)
        accuracy = 100 * correct / len(results) if results else 0
        mean_confidence = np.mean([r["confidence"] for r in results]) if results else 0

        summary = {
            "num_samples": len(results),
            "accuracy_on_samples": accuracy,
            "mean_confidence": mean_confidence,
            "samples": results,
        }

        results_path = os.path.join(self.output_dir, "ig_results.json")
        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2)

        print("\nIntegrated Gradients Results:")
        print(f"  Accuracy on samples: {accuracy:.2f}%")
        print(f"  Mean confidence: {mean_confidence:.4f}")
        print(f"  Visualizations saved to {viz_dir}")

        return summary

    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete GLUE SST-2 experiment pipeline.

        Returns:
            Dictionary with all results
        """
        print("=" * 60)
        print("GLUE SST-2 BERT Integrated Gradients Experiment Pipeline")
        print("=" * 60)

        start_time = time.time()

        print("\n[Step 1/4] Preparing data...")
        self.prepare_data()

        print("\n[Step 2/4] Fine-tuning BERT...")
        self.train()

        print("\n[Step 3/4] Final evaluation...")
        _, val_acc = self.evaluate()
        print(f"Final Validation Accuracy: {val_acc:.2f}%")

        print("\n[Step 4/4] Generating Integrated Gradients explanations...")
        ig_results = self.generate_integrated_gradients(num_samples=10, n_steps=30)

        total_time = time.time() - start_time

        results = {
            "model_name": self.model_name,
            "config": {
                k: str(v)
                if not isinstance(v, (int, float, bool, str, list, dict, type(None)))
                else v
                for k, v in self.config.items()
            },
            "training_history": self.history,
            "final_val_accuracy": val_acc,
            "ig_results": ig_results,
            "total_time_seconds": total_time,
        }

        final_results_path = os.path.join(self.output_dir, "experiment_results.json")
        with open(final_results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print("\n" + "=" * 60)
        print(f"Pipeline completed in {total_time:.1f} seconds")
        print(f"Results saved to {self.output_dir}")
        print("=" * 60)

        return results
