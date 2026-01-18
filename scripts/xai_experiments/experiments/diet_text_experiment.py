"""DiET for Text: Discriminative Feature Attribution for Transformer Models.

This module adapts DiET principles for text classification with BERT,
creating discriminative token attributions that can be compared with
Integrated Gradients.

Supports resumable training via checkpoint snapshots.

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
import numpy as np
from typing import Dict, Any, Tuple, List
from tqdm import tqdm

try:
    from ..utils.checkpointing import CheckpointManager
except ImportError:
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.checkpointing import CheckpointManager


class TokenMaskDataset(Dataset):
    """Dataset for DiET text training with token masks."""

    def __init__(self, input_ids, attention_mask, labels, predictions):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.predictions = predictions

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            idx,
            self.input_ids[idx],
            self.attention_mask[idx],
            self.labels[idx],
            self.predictions[idx],
        )


class DiETTextExplainer:
    """DiET-based explainer for transformer text models.

    Adapts DiET principles to learn token-level importance masks
    for discriminative attribution in text classification.
    """

    def __init__(self, model, device: str = "cuda", max_length: int = 128):
        """Initialize DiET text explainer.

        Args:
            model: BERT model wrapper with embedding access
            device: Device for computation
            max_length: Maximum sequence length
        """
        self.model = model
        self.device = device
        self.max_length = max_length
        self.model.eval_mode()

    def get_predictions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        batch_size: int = 16,
    ) -> torch.Tensor:
        """Get model predictions for all samples.

        Args:
            input_ids: Token IDs tensor
            attention_mask: Attention mask tensor
            batch_size: Batch size

        Returns:
            Softmax predictions
        """
        self.model.eval_mode()
        num_samples = len(input_ids)
        predictions = []

        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                batch_ids = input_ids[i : i + batch_size].to(self.device)
                batch_mask = attention_mask[i : i + batch_size].to(self.device)

                outputs = self.model(batch_ids, batch_mask)
                preds = F.softmax(outputs, dim=1).cpu()
                predictions.append(preds)

        return torch.cat(predictions, dim=0)

    def train_token_mask(
        self,
        token_mask: torch.Tensor,
        data_loader: DataLoader,
        mask_optimizer: optim.Optimizer,
        sparsity_weight: float,
    ) -> Dict[str, float]:
        """Train token importance mask.

        For text, we learn a mask over token embeddings rather than pixels.

        Args:
            token_mask: Learnable mask (N, max_length)
            data_loader: DataLoader
            mask_optimizer: Optimizer
            sparsity_weight: Sparsity regularization weight

        Returns:
            Training metrics
        """
        token_mask.requires_grad_(True)
        self.model.eval_mode()

        total_loss = 0
        total_faithful_acc = 0
        num_samples = 0

        for idx, input_ids, attn_mask, labels, pred_original in data_loader:
            input_ids = input_ids.to(self.device)
            attn_mask = attn_mask.to(self.device)
            pred_original = pred_original.to(self.device)

            embeddings = self.model.get_embeddings(input_ids)

            batch_mask = token_mask[idx].unsqueeze(-1).to(self.device)
            masked_embeddings = embeddings * batch_mask

            pred_masked = F.softmax(
                self.model.forward_with_embeddings(masked_embeddings, attn_mask), dim=1
            )

            pred_full = F.softmax(self.model(input_ids, attn_mask), dim=1)

            t1 = torch.norm(pred_original - pred_full, p=1, dim=1).sum()
            t2 = torch.norm(pred_full - pred_masked, p=1, dim=1).sum()

            active_tokens_mask = attn_mask.float()
            masked_token_mask = token_mask[idx].to(self.device) * active_tokens_mask

            # Sparsity = (Sum of mask values) / (Number of non-padding tokens)
            sparsity = masked_token_mask.sum(1) / active_tokens_mask.sum(1).clamp(min=1)
            sparsity_loss = sparsity.mean()

            loss = (sparsity_weight * sparsity_loss + t1 + t2) / len(idx)

            mask_optimizer.zero_grad()
            loss.backward()
            mask_optimizer.step()

            with torch.no_grad():
                token_mask.clamp_(0, 1)

            with torch.no_grad():
                faithful_acc = (
                    (pred_original.argmax(1) == pred_masked.argmax(1)).float().sum()
                )

            total_loss += loss.item() * len(idx)
            total_faithful_acc += faithful_acc.item()
            num_samples += len(labels)

        return {
            "loss": total_loss / num_samples,
            "faithful_acc": total_faithful_acc / num_samples,
        }

    def get_token_attributions(
        self, text: str, token_mask: torch.Tensor, sample_idx: int
    ) -> Tuple[np.ndarray, List[str]]:
        """Get token attributions from learned mask.

        Args:
            text: Input text
            token_mask: Learned mask tensor
            sample_idx: Index in the mask tensor

        Returns:
            Tuple of (attributions, tokens)
        """

        input_ids, _ = self.model.encode_text(text, self.max_length)
        tokens = self.model.decode_tokens(input_ids)

        mask_values = token_mask[sample_idx].detach().cpu().numpy()

        seq_len = len(tokens)
        attributions = mask_values[:seq_len]

        return attributions, tokens


class DiETTextExperiment:
    """DiET experiment for text classification with BERT.

    Supports multiple datasets: SST-2, IMDB, AG News.
    Compares DiET token attributions with Integrated Gradients.
    Supports resumable training via checkpoint snapshots.
    """

    # Dataset configurations: {dataset_name: (num_labels, class_names, default_max_length)}
    DATASET_CONFIGS = {
        "sst2": (2, ["negative", "positive"], 128),
        "imdb": (2, ["negative", "positive"], 256),
        "ag_news": (4, ["World", "Sports", "Business", "Sci/Tech"], 128),
    }

    def __init__(self, config: Dict[str, Any]):
        """Initialize DiET text experiment.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.data_dir = config.get("data_dir", "./data")
        self.output_dir = config.get(
            "output_dir", "./outputs/xai_experiments/diet_text"
        )
        os.makedirs(self.output_dir, exist_ok=True)

        self.model_name = config.get("model_name", "bert-base-uncased")
        self.model = None

        # Get dataset name and configuration
        self.dataset_name = config.get("dataset", "sst2").lower()
        if self.dataset_name not in self.DATASET_CONFIGS:
            raise ValueError(
                f"Unsupported dataset: {self.dataset_name}. "
                f"Supported: {list(self.DATASET_CONFIGS.keys())}"
            )

        self.num_labels, self.class_labels, default_max_length = self.DATASET_CONFIGS[
            self.dataset_name
        ]

        # Batch size configuration for memory optimization
        self.batch_size = config.get("batch_size", 16)

        # Determine if low VRAM mode is enabled
        # Check explicit flag first, then infer from batch size
        self.low_vram = config.get("low_vram", self.batch_size <= 8)

        # Use config max_length or dataset default, with low VRAM adjustment
        config_max_length = config.get("max_length", default_max_length)
        # For IMDB on low VRAM, reduce max_length to save memory
        if self.dataset_name == "imdb" and config_max_length > 128 and self.low_vram:
            self.max_length = min(config_max_length, 128)
        else:
            self.max_length = config_max_length

        self.train_input_ids = None
        self.train_attention_mask = None
        self.train_labels = None
        self.test_texts = None
        self.test_labels = None

        self.results = {}

        # Initialize checkpoint manager for resumable training
        checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
        self.experiment_name = config.get(
            "experiment_name", f"diet_text_{self.dataset_name}"
        )

    def _init_model(self):
        """Initialize BERT model with correct number of labels for the dataset."""
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
                model_name=self.model_name,
                num_labels=self.num_labels,
                device=self.device,
            )

    def prepare_data(self, max_samples: int = 2000) -> None:
        """Prepare dataset for training.

        Loads the dataset specified in config (sst2, imdb, or ag_news).

        Args:
            max_samples: Maximum training samples
        """
        print(f"Loading {self.dataset_name.upper()} dataset...")
        from datasets import load_dataset

        # Load dataset based on configuration
        if self.dataset_name == "sst2":
            dataset = load_dataset("glue", "sst2", cache_dir=self.data_dir)
            train_texts = dataset["train"]["sentence"][:max_samples]
            train_labels = dataset["train"]["label"][:max_samples]
            val_texts = dataset["validation"]["sentence"][:200]
            val_labels = dataset["validation"]["label"][:200]
        elif self.dataset_name == "imdb":
            dataset = load_dataset("imdb", cache_dir=self.data_dir)
            train_texts = dataset["train"]["text"][:max_samples]
            train_labels = dataset["train"]["label"][:max_samples]
            # Use subset of test set for validation
            val_texts = dataset["test"]["text"][:200]
            val_labels = dataset["test"]["label"][:200]
        elif self.dataset_name == "ag_news":
            dataset = load_dataset("ag_news", cache_dir=self.data_dir)
            train_texts = dataset["train"]["text"][:max_samples]
            train_labels = dataset["train"]["label"][:max_samples]
            val_texts = dataset["test"]["text"][:200]
            val_labels = dataset["test"]["label"][:200]
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        self._init_model()

        print("Tokenizing training data...")
        train_input_ids = []
        train_attention_masks = []

        for text in tqdm(train_texts, desc="Tokenizing"):
            input_ids, attn_mask = self.model.encode_text(text, self.max_length)
            train_input_ids.append(input_ids.squeeze(0))
            train_attention_masks.append(attn_mask.squeeze(0))

        self.train_input_ids = torch.stack(train_input_ids)
        self.train_attention_mask = torch.stack(train_attention_masks)
        self.train_labels = torch.tensor(train_labels)

        self.test_texts = val_texts
        self.test_labels = val_labels

        print(f"Training samples: {len(self.train_labels)}")
        print(f"Test samples: {len(self.test_texts)}")
        print(f"Max sequence length: {self.max_length}")

    def train_baseline(
        self, epochs: int = 2, save_checkpoint_every: int = 1
    ) -> Dict[str, Any]:
        """Train baseline BERT model with checkpoint support.

        Training can be resumed from the last checkpoint if interrupted.

        Args:
            epochs: Training epochs
            save_checkpoint_every: Save checkpoint every N epochs (default: 1)

        Returns:
            Training history
        """
        checkpoint_name = f"{self.experiment_name}_baseline"
        start_epoch = 0

        print("\nFine-tuning BERT baseline...")

        self._init_model()

        train_dataset = torch.utils.data.TensorDataset(
            self.train_input_ids, self.train_attention_mask, self.train_labels
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.get_parameters(), lr=2e-5)

        history = {"train_loss": [], "train_acc": []}

        # Check for existing checkpoint
        if self.checkpoint_manager.has_checkpoint(checkpoint_name):
            print("Found checkpoint, resuming training...")
            checkpoint = self.checkpoint_manager.load_checkpoint(
                checkpoint_name, self.device
            )
            if checkpoint:
                self.model.load_state_dict_from_checkpoint(
                    checkpoint["model_state_dict"]
                )
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                start_epoch = checkpoint["epoch"] + 1
                history = checkpoint.get("extra_state", {}).get("history", history)
                print(f"Resuming from epoch {start_epoch}")

        for epoch in range(start_epoch, epochs):
            self.model.train_mode()
            train_loss = 0
            correct = 0
            total = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            for input_ids, attn_mask, labels in pbar:
                input_ids = input_ids.to(self.device)
                attn_mask = attn_mask.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(input_ids, attn_mask)
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

            history["train_loss"].append(train_loss / len(train_loader))
            history["train_acc"].append(100 * correct / total)

            # Save checkpoint
            if (epoch + 1) % save_checkpoint_every == 0 or epoch == epochs - 1:
                self.checkpoint_manager.save_checkpoint(
                    checkpoint_name,
                    epoch=epoch,
                    model_state_dict=self.model.get_state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    metrics={"train_acc": history["train_acc"][-1]},
                    extra_state={"history": history},
                )

        baseline_path = os.path.join(self.output_dir, "bert_baseline")
        self.model.save_model(baseline_path)

        # Clean up checkpoint after successful completion
        self.checkpoint_manager.delete_checkpoint(checkpoint_name)

        val_acc = self._evaluate_model()

        self.results["baseline"] = {"history": history, "val_acc": val_acc}

        print(f"Baseline validation accuracy: {val_acc:.2f}%")

        return history

    def _evaluate_model(self) -> float:
        """Evaluate on validation set."""
        self.model.eval_mode()
        correct = 0
        
        eval_texts = self.test_texts[:100]
        eval_labels = self.test_labels[:100]
        num_eval = len(eval_texts)

        for text, label in zip(eval_texts, eval_labels):
            input_ids, attn_mask = self.model.encode_text(text, self.max_length)
            input_ids = input_ids.to(self.device)
            attn_mask = attn_mask.to(self.device)

            with torch.no_grad():
                output = self.model(input_ids, attn_mask)
                pred = output.argmax(1).item()

            if pred == label:
                correct += 1

        return 100 * correct / num_eval

    def load_baseline(self, model_path: str = None) -> bool:
        """Load a previously trained baseline BERT model.

        Args:
            model_path: Path to the model checkpoint. If None, uses default path.

        Returns:
            True if model was loaded successfully, False otherwise
        """
        self._init_model()

        if model_path is None:
            model_path = os.path.join(self.output_dir, "bert_baseline")

        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            return False

        try:
            self.model.load_model(model_path)
            self.model.eval_mode()
            val_acc = self._evaluate_model()
            print(f"Loaded baseline model from {model_path}")
            print(f"Validation accuracy: {val_acc:.2f}%")

            self.results["baseline"] = {"val_acc": val_acc, "loaded_from": model_path}
            return True
        except Exception as e:
            print(f"Failed to load model from {model_path}: {e}")
            return False

    def run_diet(self, rounding_steps: int = 2) -> Dict[str, Any]:
        """Run DiET for token attribution learning.

        Args:
            rounding_steps: Number of distillation steps

        Returns:
            DiET results
        """
        print("\n" + "=" * 50)
        print("Running DiET for Text")
        print("=" * 50)

        diet = DiETTextExplainer(self.model, self.device, self.max_length)

        print("Getting baseline predictions...")
        train_preds = diet.get_predictions(
            self.train_input_ids, self.train_attention_mask
        )

        diet_dataset = TokenMaskDataset(
            self.train_input_ids,
            self.train_attention_mask,
            self.train_labels,
            train_preds,
        )
        diet_loader = DataLoader(
            diet_dataset, batch_size=self.batch_size, shuffle=True
        )

        token_mask = torch.ones((len(train_preds), self.max_length), requires_grad=True)
        mask_optimizer = optim.Adam([token_mask], lr=0.1)

        sparsity_weights = [
            1.0 - r * (0.8 / rounding_steps) for r in range(rounding_steps)
        ]
        rounding_scheme = [
            0.3 - r * (0.3 / rounding_steps) for r in range(rounding_steps)
        ]

        diet_history = []

        for step in range(rounding_steps):
            print(f"\n--- DiET Step {step + 1}/{rounding_steps} ---")

            prev_loss = float("inf")
            for i in range(30):
                metrics = diet.train_token_mask(
                    token_mask, diet_loader, mask_optimizer, sparsity_weights[step]
                )

                if abs(metrics["loss"] - prev_loss) < 0.001:
                    break
                prev_loss = metrics["loss"]

                if i % 10 == 0:
                    print(
                        f"  Iter {i}: loss={metrics['loss']:.4f}, faithful_acc={metrics['faithful_acc']:.4f}"
                    )

            with torch.no_grad():
                token_mask.copy_(
                    torch.round(token_mask + rounding_scheme[step]).clamp(0, 1)
                )

            diet_history.append(metrics)

        mask_path = os.path.join(self.output_dir, "diet_token_mask.pt")
        torch.save(token_mask.detach(), mask_path)

        self.diet_token_mask = token_mask.detach()

        self.results["diet"] = {"history": diet_history}

        return self.results["diet"]

    def compare_with_ig(self, num_samples: int = 10) -> Dict[str, Any]:
        """Compare DiET with Integrated Gradients.

        Args:
            num_samples: Number of samples for comparison

        Returns:
            Comparison results
        """
        print("\n" + "=" * 50)
        print("Comparing DiET vs Integrated Gradients")
        print("=" * 50)

        try:
            from ..explainers.integrated_gradients import IntegratedGradientsText
        except ImportError:
            import sys

            sys.path.insert(
                0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            from explainers.integrated_gradients import IntegratedGradientsText

        num_samples = min(num_samples, len(self.train_input_ids), len(self.diet_token_mask))
        ig = IntegratedGradientsText(self.model, self.device)
        diet = DiETTextExplainer(self.model, self.device, self.max_length)

        comparison_results = []
        with torch.no_grad():
            for i in tqdm(range(num_samples), desc="Comparing methods"):
                input_ids = self.train_input_ids[i].unsqueeze(0).to(self.device)
                tokens = self.model.decode_tokens(input_ids[0])
                text = " ".join(tokens) 
                label = self.train_labels[i].item()

                try:
                    ig_attrs, _, pred_class, confidence = ig.compute_attributions(text, n_steps=20)

                    mask_values = self.diet_token_mask[i].cpu().numpy()
                    diet_attrs = mask_values[:len(tokens)]
                    diet_tokens = tokens

                    special_tokens = {"[PAD]", "[CLS]", "[SEP]", "[UNK]", "<pad>", "<s>", "</s>"}
                
                    valid_indices = [j for j, t in enumerate(tokens) if t not in special_tokens]
                    if not valid_indices: continue

                    # Adjust k based on sentence length
                    k = max(1, min(5, len(valid_indices)))

                    # Sort IG
                    ig_valid = sorted([(j, np.abs(ig_attrs[j])) for j in valid_indices if j < len(ig_attrs)], 
                                    key=lambda x: x[1], reverse=True)
                    ig_top_k = set([x[0] for x in ig_valid[:k]])

                    # Sort DiET
                    diet_valid = sorted([(j, diet_attrs[j]) for j in valid_indices if j < len(diet_attrs)], 
                                    key=lambda x: x[1], reverse=True)
                    diet_top_k = set([x[0] for x in diet_valid[:k]])

                    overlap = len(ig_top_k & diet_top_k) / k

                    comparison_results.append(
                        {
                            "text": text[:50] + "...",
                            "true_label": label,
                            "pred_class": pred_class,
                            "confidence": confidence,
                            "top_k_overlap": overlap,
                            "ig_top_tokens": [
                                tokens[j] for j in ig_top_k if j < len(tokens)
                            ],
                            "diet_top_tokens": [
                                diet_tokens[j] for j in diet_top_k if j < len(diet_tokens)
                            ],
                        }
                    )

                except Exception as e:
                    print(f"Error on sample {i}: {e}")
                    continue

        mean_overlap = np.mean([r["top_k_overlap"] for r in comparison_results])

        comparison = {
            "samples": comparison_results,
            "mean_top_k_overlap": mean_overlap,
            "num_samples": len(comparison_results),
        }

        self.results["comparison"] = comparison

        print("\nComparison Results:")
        print(f"  Mean top-k token overlap: {mean_overlap:.4f}")
        print("  (1.0 = perfect agreement, 0.0 = no overlap)")

        self._save_text_comparison(comparison_results[:5])

        return comparison

    def _save_text_comparison(self, samples: List[Dict]) -> None:
        """Save text comparison visualizations."""
        viz_dir = os.path.join(self.output_dir, "comparison_visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        html = """
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; padding: 20px; }
                .sample { margin-bottom: 30px; border-bottom: 1px solid 
                .method { margin: 10px 0; }
                .label { font-weight: bold; }
                .tokens { display: flex; flex-wrap: wrap; gap: 5px; }
                .token { background: 
                .top-token { background: 
            </style>
        </head>
        <body>
            <h1>DiET vs Integrated Gradients Comparison</h1>
        """

        for i, sample in enumerate(samples):
            label_name = self.class_labels[sample["true_label"]]
            pred_name = self.class_labels[sample["pred_class"]]

            html += f"""
            <div class="sample">
                <h3>Sample {i + 1}</h3>
                <p><span class="label">Text:</span> {sample["text"]}</p>
                <p><span class="label">True Label:</span> {label_name}</p>
                <p><span class="label">Predicted:</span> {pred_name} (conf: {sample["confidence"]:.2%})</p>
                <p><span class="label">Top-k Overlap:</span> {sample["top_k_overlap"]:.2%}</p>
                
                <div class="method">
                    <p><span class="label">IG Top Tokens:</span></p>
                    <div class="tokens">
            """
            for token in sample["ig_top_tokens"]:
                html += f'<span class="token top-token">{token}</span>'
            html += """
                    </div>
                </div>
                
                <div class="method">
                    <p><span class="label">DiET Top Tokens:</span></p>
                    <div class="tokens">
            """
            for token in sample["diet_top_tokens"]:
                html += f'<span class="token top-token">{token}</span>'
            html += """
                    </div>
                </div>
            </div>
            """

        html += "</body></html>"

        with open(os.path.join(viz_dir, "text_comparison.html"), "w") as f:
            f.write(html)

        print(f"Text comparison saved to {viz_dir}")

    def run_full_experiment(self, skip_training: bool = False) -> Dict[str, Any]:
        """Run complete DiET text experiment.

        Args:
            skip_training: If True, try to load previously trained baseline model
                          instead of training from scratch

        Returns:
            All results
        """
        print("=" * 60)
        print("DiET Text Experiment: Token Attribution")
        print("=" * 60)

        start_time = time.time()

        print("\n[Step 1/4] Preparing data...")
        max_samples = self.config.get("max_samples", 2000)
        self.prepare_data(max_samples)

        if skip_training:
            print("\n[Step 2/4] Loading baseline BERT...")
            if not self.load_baseline():
                print("Failed to load saved model. Training from scratch...")
                epochs = self.config.get("epochs", 2)
                self.train_baseline(epochs)
        else:
            print("\n[Step 2/4] Training baseline BERT...")
            epochs = self.config.get("epochs", 2)
            self.train_baseline(epochs)

        print("\n[Step 3/4] Running DiET distillation...")
        rounding_steps = self.config.get("rounding_steps", 2)
        self.run_diet(rounding_steps)

        print("\n[Step 4/4] Comparing DiET vs IG...")
        comparison_samples = self.config.get("comparison_samples", 10)
        self.compare_with_ig(comparison_samples)

        total_time = time.time() - start_time
        self.results["total_time_seconds"] = total_time

        results_path = os.path.join(self.output_dir, "diet_text_results.json")

        def make_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            return obj

        with open(results_path, "w") as f:
            json.dump(make_serializable(self.results), f, indent=2)

        print("\n" + "=" * 60)
        print("DiET Text Experiment Summary")
        print("=" * 60)
        print(f"Baseline Val Accuracy: {self.results['baseline']['val_acc']:.2f}%")
        print(
            f"DiET-IG Top-k Overlap: {self.results['comparison']['mean_top_k_overlap']:.4f}"
        )
        print(f"\nTotal time: {total_time:.1f} seconds")
        print(f"Results saved to {results_path}")
        print("=" * 60)

        return self.results
