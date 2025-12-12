"""DiET for Text: Discriminative Feature Attribution for Transformer Models.

This module adapts DiET principles for text classification with BERT,
creating discriminative token attributions that can be compared with
Integrated Gradients.

Reference: Bhalla et al., "Discriminative Feature Attributions", NeurIPS 2023
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
from typing import Optional, Dict, Any, Tuple, List
from tqdm import tqdm

# Use non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
            self.predictions[idx]
        )


class DiETTextExplainer:
    """DiET-based explainer for transformer text models.
    
    Adapts DiET principles to learn token-level importance masks
    for discriminative attribution in text classification.
    """
    
    def __init__(
        self,
        model,
        device: str = "cuda",
        max_length: int = 128
    ):
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
        batch_size: int = 16
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
                batch_ids = input_ids[i:i+batch_size].to(self.device)
                batch_mask = attention_mask[i:i+batch_size].to(self.device)
                
                outputs = self.model(batch_ids, batch_mask)
                preds = F.softmax(outputs, dim=1).cpu()
                predictions.append(preds)
        
        return torch.cat(predictions, dim=0)
    
    def train_token_mask(
        self,
        token_mask: torch.Tensor,
        data_loader: DataLoader,
        mask_optimizer: optim.Optimizer,
        sparsity_weight: float
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
            
            # Get embeddings
            embeddings = self.model.get_embeddings(input_ids)
            
            # Apply mask to embeddings
            batch_mask = token_mask[idx].unsqueeze(-1).to(self.device)  # (B, L, 1)
            masked_embeddings = embeddings * batch_mask
            
            # Get predictions on masked embeddings
            pred_masked = F.softmax(
                self.model.forward_with_embeddings(masked_embeddings, attn_mask),
                dim=1
            )
            
            # Get predictions on full embeddings
            pred_full = F.softmax(self.model(input_ids, attn_mask), dim=1)
            
            # Faithfulness loss: masked predictions should match original
            t1 = torch.norm(pred_original - pred_full, p=1, dim=1).sum()
            t2 = torch.norm(pred_full - pred_masked, p=1, dim=1).sum()
            
            # Sparsity: encourage focusing on fewer tokens
            # Weight by attention mask to only count real tokens
            sparsity = (token_mask[idx].to(self.device) * attn_mask.float()).sum(1)
            sparsity = sparsity / attn_mask.float().sum(1)  # Normalize by sequence length
            sparsity = sparsity.sum()
            
            loss = (sparsity_weight * sparsity + t1 + t2) / len(idx)
            
            mask_optimizer.zero_grad()
            loss.backward()
            mask_optimizer.step()
            
            # Clamp mask
            with torch.no_grad():
                token_mask.clamp_(0, 1)
            
            # Metrics
            with torch.no_grad():
                faithful_acc = (pred_original.argmax(1) == pred_masked.argmax(1)).float().sum()
            
            total_loss += loss.item() * len(idx)
            total_faithful_acc += faithful_acc.item()
            num_samples += len(labels)
        
        return {
            "loss": total_loss / num_samples,
            "faithful_acc": total_faithful_acc / num_samples
        }
    
    def get_token_attributions(
        self,
        text: str,
        token_mask: torch.Tensor,
        sample_idx: int
    ) -> Tuple[np.ndarray, List[str]]:
        """Get token attributions from learned mask.
        
        Args:
            text: Input text
            token_mask: Learned mask tensor
            sample_idx: Index in the mask tensor
            
        Returns:
            Tuple of (attributions, tokens)
        """
        # Encode text
        input_ids, _ = self.model.encode_text(text, self.max_length)
        tokens = self.model.decode_tokens(input_ids)
        
        # Get mask values
        mask_values = token_mask[sample_idx].detach().cpu().numpy()
        
        # Only return values for actual tokens (not padding)
        seq_len = len(tokens)
        attributions = mask_values[:seq_len]
        
        return attributions, tokens


class DiETTextExperiment:
    """DiET experiment for text classification (SST-2) with BERT.
    
    Compares DiET token attributions with Integrated Gradients.
    """
    
    SST2_LABELS = ["negative", "positive"]
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize DiET text experiment.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        self.data_dir = config.get("data_dir", "./data/glue_sst2")
        self.output_dir = config.get("output_dir", "./outputs/xai_experiments/diet_text")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.model_name = config.get("model_name", "bert-base-uncased")
        self.max_length = config.get("max_length", 128)
        self.model = None
        
        # Data
        self.train_input_ids = None
        self.train_attention_mask = None
        self.train_labels = None
        self.test_texts = None
        self.test_labels = None
        
        self.results = {}
    
    def _init_model(self):
        """Initialize BERT model."""
        if self.model is None:
            try:
                from ..models.transformer import BertForSequenceClassificationWithIG
            except ImportError:
                import sys
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from models.transformer import BertForSequenceClassificationWithIG
            
            self.model = BertForSequenceClassificationWithIG(
                model_name=self.model_name,
                num_labels=2,
                device=self.device
            )
    
    def prepare_data(self, max_samples: int = 2000) -> None:
        """Prepare SST-2 data.
        
        Args:
            max_samples: Maximum training samples
        """
        print("Loading SST-2 dataset...")
        from datasets import load_dataset
        
        dataset = load_dataset("glue", "sst2", cache_dir=self.data_dir)
        self._init_model()
        
        # Prepare training data
        train_texts = dataset["train"]["sentence"][:max_samples]
        train_labels = dataset["train"]["label"][:max_samples]
        
        # Tokenize
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
        
        # Test data (keep as text for IG comparison)
        self.test_texts = dataset["validation"]["sentence"][:200]
        self.test_labels = dataset["validation"]["label"][:200]
        
        print(f"Training samples: {len(self.train_labels)}")
        print(f"Test samples: {len(self.test_texts)}")
    
    def train_baseline(self, epochs: int = 2) -> Dict[str, Any]:
        """Train baseline BERT model.
        
        Args:
            epochs: Training epochs
            
        Returns:
            Training history
        """
        print(f"\nFine-tuning BERT baseline...")
        
        self._init_model()
        
        # Create dataloader
        train_dataset = torch.utils.data.TensorDataset(
            self.train_input_ids,
            self.train_attention_mask,
            self.train_labels
        )
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.get_parameters(), lr=2e-5)
        
        history = {"train_loss": [], "train_acc": []}
        
        for epoch in range(epochs):
            self.model.train_mode()
            train_loss = 0
            correct = 0
            total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
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
                
                pbar.set_postfix({"loss": f"{train_loss/(pbar.n+1):.4f}", "acc": f"{100*correct/total:.2f}%"})
            
            history["train_loss"].append(train_loss / len(train_loader))
            history["train_acc"].append(100 * correct / total)
        
        # Save baseline
        baseline_path = os.path.join(self.output_dir, "bert_baseline")
        self.model.save_model(baseline_path)
        
        # Evaluate
        val_acc = self._evaluate_model()
        
        self.results["baseline"] = {
            "history": history,
            "val_acc": val_acc
        }
        
        print(f"Baseline validation accuracy: {val_acc:.2f}%")
        
        return history
    
    def _evaluate_model(self) -> float:
        """Evaluate on validation set."""
        self.model.eval_mode()
        correct = 0
        
        for i, (text, label) in enumerate(zip(self.test_texts[:100], self.test_labels[:100])):
            input_ids, attn_mask = self.model.encode_text(text, self.max_length)
            input_ids = input_ids.to(self.device)
            attn_mask = attn_mask.to(self.device)
            
            with torch.no_grad():
                output = self.model(input_ids, attn_mask)
                pred = output.argmax(1).item()
            
            if pred == label:
                correct += 1
        
        return 100 * correct / 100
    
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
            
            self.results["baseline"] = {
                "val_acc": val_acc,
                "loaded_from": model_path
            }
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
        
        # Initialize DiET
        diet = DiETTextExplainer(self.model, self.device, self.max_length)
        
        # Get predictions
        print("Getting baseline predictions...")
        train_preds = diet.get_predictions(self.train_input_ids, self.train_attention_mask)
        
        # Create dataset
        diet_dataset = TokenMaskDataset(
            self.train_input_ids,
            self.train_attention_mask,
            self.train_labels,
            train_preds
        )
        diet_loader = DataLoader(diet_dataset, batch_size=16, shuffle=True)
        
        # Initialize token mask
        token_mask = torch.ones((len(train_preds), self.max_length), requires_grad=True)
        mask_optimizer = optim.Adam([token_mask], lr=0.1)
        
        # Training
        sparsity_weights = [1.0 - r * (0.8 / rounding_steps) for r in range(rounding_steps)]
        rounding_scheme = [0.3 - r * (0.3 / rounding_steps) for r in range(rounding_steps)]
        
        diet_history = []
        
        for step in range(rounding_steps):
            print(f"\n--- DiET Step {step+1}/{rounding_steps} ---")
            
            # Train mask
            prev_loss = float('inf')
            for i in range(30):
                metrics = diet.train_token_mask(
                    token_mask, diet_loader, mask_optimizer, sparsity_weights[step]
                )
                
                if abs(metrics["loss"] - prev_loss) < 0.001:
                    break
                prev_loss = metrics["loss"]
                
                if i % 10 == 0:
                    print(f"  Iter {i}: loss={metrics['loss']:.4f}, faithful_acc={metrics['faithful_acc']:.4f}")
            
            # Round mask
            with torch.no_grad():
                token_mask.copy_(torch.round(token_mask + rounding_scheme[step]).clamp(0, 1))
            
            diet_history.append(metrics)
        
        # Save mask
        mask_path = os.path.join(self.output_dir, "diet_token_mask.pt")
        torch.save(token_mask.detach(), mask_path)
        
        self.diet_token_mask = token_mask.detach()
        
        self.results["diet"] = {
            "history": diet_history
        }
        
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
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from explainers.integrated_gradients import IntegratedGradientsText
        
        ig = IntegratedGradientsText(self.model, self.device)
        diet = DiETTextExplainer(self.model, self.device, self.max_length)
        
        comparison_results = []
        
        for i in tqdm(range(num_samples), desc="Comparing methods"):
            text = self.test_texts[i]
            label = self.test_labels[i]
            
            try:
                # Get IG attributions
                ig_attrs, tokens, pred_class, confidence = ig.compute_attributions(
                    text, n_steps=20
                )
                
                # Get DiET attributions (using sample from training mask as proxy)
                sample_idx = i % len(self.diet_token_mask)
                diet_attrs, diet_tokens = diet.get_token_attributions(
                    text, self.diet_token_mask, sample_idx
                )
                
                # Compute overlap/agreement metrics
                # Find top-k tokens for each method
                k = min(5, len(tokens) // 2)
                
                ig_top_k = set(np.argsort(np.abs(ig_attrs))[-k:])
                diet_top_k = set(np.argsort(diet_attrs[:len(ig_attrs)])[-k:])
                
                overlap = len(ig_top_k & diet_top_k) / k
                
                comparison_results.append({
                    "text": text[:50] + "...",
                    "true_label": label,
                    "pred_class": pred_class,
                    "confidence": confidence,
                    "top_k_overlap": overlap,
                    "ig_top_tokens": [tokens[j] for j in ig_top_k if j < len(tokens)],
                    "diet_top_tokens": [diet_tokens[j] for j in diet_top_k if j < len(diet_tokens)]
                })
                
            except Exception as e:
                print(f"Error on sample {i}: {e}")
                continue
        
        mean_overlap = np.mean([r["top_k_overlap"] for r in comparison_results])
        
        comparison = {
            "samples": comparison_results,
            "mean_top_k_overlap": mean_overlap,
            "num_samples": len(comparison_results)
        }
        
        self.results["comparison"] = comparison
        
        print(f"\nComparison Results:")
        print(f"  Mean top-k token overlap: {mean_overlap:.4f}")
        print(f"  (1.0 = perfect agreement, 0.0 = no overlap)")
        
        # Save visualizations
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
                .sample { margin-bottom: 30px; border-bottom: 1px solid #ccc; padding-bottom: 20px; }
                .method { margin: 10px 0; }
                .label { font-weight: bold; }
                .tokens { display: flex; flex-wrap: wrap; gap: 5px; }
                .token { background: #e0e0e0; padding: 2px 6px; border-radius: 3px; }
                .top-token { background: #4CAF50; color: white; }
            </style>
        </head>
        <body>
            <h1>DiET vs Integrated Gradients Comparison</h1>
        """
        
        for i, sample in enumerate(samples):
            label_name = self.SST2_LABELS[sample["true_label"]]
            pred_name = self.SST2_LABELS[sample["pred_class"]]
            
            html += f"""
            <div class="sample">
                <h3>Sample {i+1}</h3>
                <p><span class="label">Text:</span> {sample['text']}</p>
                <p><span class="label">True Label:</span> {label_name}</p>
                <p><span class="label">Predicted:</span> {pred_name} (conf: {sample['confidence']:.2%})</p>
                <p><span class="label">Top-k Overlap:</span> {sample['top_k_overlap']:.2%}</p>
                
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
        
        # Step 1: Prepare data
        print("\n[Step 1/4] Preparing data...")
        max_samples = self.config.get("max_samples", 2000)
        self.prepare_data(max_samples)
        
        # Step 2: Train or load baseline
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
        
        # Step 3: Run DiET
        print("\n[Step 3/4] Running DiET distillation...")
        rounding_steps = self.config.get("rounding_steps", 2)
        self.run_diet(rounding_steps)
        
        # Step 4: Compare methods
        print("\n[Step 4/4] Comparing DiET vs IG...")
        comparison_samples = self.config.get("comparison_samples", 10)
        self.compare_with_ig(comparison_samples)
        
        total_time = time.time() - start_time
        self.results["total_time_seconds"] = total_time
        
        # Save results
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
        print(f"DiET-IG Top-k Overlap: {self.results['comparison']['mean_top_k_overlap']:.4f}")
        print(f"\nTotal time: {total_time:.1f} seconds")
        print(f"Results saved to {results_path}")
        print("=" * 60)
        
        return self.results
