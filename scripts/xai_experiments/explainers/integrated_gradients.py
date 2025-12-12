"""Integrated Gradients Implementation for Model Explainability.

This module implements Integrated Gradients for both image and text models.
Integrated Gradients is a gradient-based attribution method that satisfies
several desirable axioms including sensitivity and implementation invariance.

Reference: Sundararajan et al., "Axiomatic Attribution for Deep Networks", ICML 2017
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Callable

# Use non-interactive backend to avoid TCL/Tk errors in headless environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class IntegratedGradients:
    """Integrated Gradients for image model attribution.
    
    This implementation computes attributions by integrating gradients along
    a path from a baseline (usually all zeros) to the input image.
    
    Attributes:
        model: The model to explain
        device: Device to run computations on
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda",
        baseline_type: str = "zeros"
    ):
        """Initialize Integrated Gradients.
        
        Args:
            model: Model to compute attributions for
            device: Device to run computations on
            baseline_type: Type of baseline ("zeros", "random", "blur")
        """
        self.model = model
        self.device = device
        self.baseline_type = baseline_type
        self.model.eval()
    
    def _get_baseline(self, input_shape: tuple) -> torch.Tensor:
        """Generate baseline input.
        
        Args:
            input_shape: Shape of the input tensor
            
        Returns:
            Baseline tensor
        """
        if self.baseline_type == "zeros":
            return torch.zeros(input_shape, device=self.device)
        elif self.baseline_type == "random":
            return torch.rand(input_shape, device=self.device)
        else:
            return torch.zeros(input_shape, device=self.device)
    
    def compute_attributions(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        n_steps: int = 50,
        baseline: Optional[torch.Tensor] = None
    ) -> Tuple[np.ndarray, int, float]:
        """Compute Integrated Gradients attributions.
        
        Args:
            input_tensor: Input tensor of shape (1, C, H, W)
            target_class: Class to compute attributions for
            n_steps: Number of integration steps
            baseline: Custom baseline tensor
            
        Returns:
            Tuple of (attributions, predicted_class, confidence)
        """
        input_tensor = input_tensor.to(self.device)
        
        if baseline is None:
            baseline = self._get_baseline(input_tensor.shape)
        
        # Forward pass to get prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            probs = F.softmax(output, dim=1)
            confidence = probs[0, target_class].item()
        
        # Compute scaled inputs along the path
        scaled_inputs = [
            baseline + (float(i) / n_steps) * (input_tensor - baseline)
            for i in range(1, n_steps + 1)
        ]
        scaled_inputs = torch.cat(scaled_inputs, dim=0)
        scaled_inputs.requires_grad_(True)
        
        # Compute gradients
        outputs = self.model(scaled_inputs)
        target_outputs = outputs[:, target_class]
        
        gradients = torch.autograd.grad(
            outputs=target_outputs,
            inputs=scaled_inputs,
            grad_outputs=torch.ones_like(target_outputs),
            create_graph=False
        )[0]
        
        # Average gradients
        avg_gradients = gradients.mean(dim=0, keepdim=True)
        
        # Compute attributions: (input - baseline) * average_gradients
        attributions = (input_tensor - baseline) * avg_gradients
        
        # Sum across channels for visualization
        attributions = attributions.sum(dim=1).squeeze()
        
        # Normalize
        attributions = attributions.detach().cpu().numpy()
        
        return attributions, target_class, confidence
    
    @staticmethod
    def visualize(
        original_image: np.ndarray,
        attributions: np.ndarray,
        predicted_class: int,
        confidence: float,
        class_names: Optional[list] = None,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """Visualize Integrated Gradients attributions.
        
        Args:
            original_image: Original image of shape (H, W, 3)
            attributions: Attribution map
            predicted_class: Predicted class index
            confidence: Prediction confidence
            class_names: List of class names
            save_path: Path to save figure
            show: Whether to display figure
        """
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV (cv2) is required for visualize. Install with: pip install opencv-python")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        
        # Attributions (positive)
        attr_resized = cv2.resize(attributions, (original_image.shape[1], original_image.shape[0]))
        pos_attr = np.clip(attr_resized, 0, None)
        axes[1].imshow(pos_attr, cmap="Reds")
        axes[1].set_title("Positive Attributions")
        axes[1].axis("off")
        
        # Overlay
        # Normalize attributions to [0, 1]
        attr_norm = (attr_resized - attr_resized.min()) / (attr_resized.max() - attr_resized.min() + 1e-8)
        
        # Create colored overlay
        cmap = plt.cm.get_cmap("RdBu_r")
        attr_colored = cmap(1 - attr_norm)[:, :, :3]
        attr_colored = (attr_colored * 255).astype(np.uint8)
        
        if original_image.max() <= 1:
            original_image = (original_image * 255).astype(np.uint8)
        
        overlaid = cv2.addWeighted(original_image, 0.6, attr_colored, 0.4, 0)
        axes[2].imshow(overlaid)
        
        class_label = class_names[predicted_class] if class_names else str(predicted_class)
        axes[2].set_title(f"Predicted: {class_label} ({confidence:.2%})")
        axes[2].axis("off")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        
        if show:
            plt.show()
        
        plt.close()


class IntegratedGradientsText:
    """Integrated Gradients for Text/Transformer models.
    
    This implementation computes attributions for BERT and similar transformer
    models by integrating gradients in the embedding space.
    """
    
    def __init__(
        self,
        model,
        device: str = "cuda"
    ):
        """Initialize Integrated Gradients for text.
        
        Args:
            model: BERT model wrapper with embedding access
            device: Device to run computations on
        """
        self.model = model
        self.device = device
        self.model.eval_mode()
    
    def compute_attributions(
        self,
        text: str,
        target_class: Optional[int] = None,
        n_steps: int = 50,
        max_length: int = 128
    ) -> Tuple[np.ndarray, list, int, float]:
        """Compute Integrated Gradients attributions for text.
        
        Args:
            text: Input text
            target_class: Class to compute attributions for
            n_steps: Number of integration steps
            max_length: Maximum sequence length
            
        Returns:
            Tuple of (attributions, tokens, predicted_class, confidence)
        """
        # Encode text
        input_ids, attention_mask = self.model.encode_text(text, max_length)
        
        # Get embeddings
        embeddings = self.model.get_embeddings(input_ids)
        
        # Create baseline (zero embeddings)
        baseline = torch.zeros_like(embeddings)
        
        # Forward pass to get prediction
        with torch.no_grad():
            output = self.model.forward_with_embeddings(embeddings, attention_mask)
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            probs = F.softmax(output, dim=1)
            confidence = probs[0, target_class].item()
        
        # Compute scaled inputs
        scaled_inputs = []
        for i in range(1, n_steps + 1):
            alpha = float(i) / n_steps
            scaled_inputs.append(baseline + alpha * (embeddings - baseline))
        
        # Compute gradients for each scaled input
        all_gradients = []
        for scaled_input in scaled_inputs:
            scaled_input = scaled_input.requires_grad_(True)
            output = self.model.forward_with_embeddings(scaled_input, attention_mask)
            target_output = output[0, target_class]
            
            gradient = torch.autograd.grad(
                outputs=target_output,
                inputs=scaled_input,
                create_graph=False
            )[0]
            all_gradients.append(gradient)
        
        # Average gradients
        avg_gradients = torch.stack(all_gradients).mean(dim=0)
        
        # Compute attributions
        attributions = (embeddings - baseline) * avg_gradients
        
        # Sum across embedding dimension to get per-token attribution
        token_attributions = attributions.sum(dim=-1).squeeze()
        
        # Get tokens
        tokens = self.model.decode_tokens(input_ids)
        
        # Apply attention mask
        token_attributions = token_attributions * attention_mask.squeeze().float()
        
        attributions_np = token_attributions.detach().cpu().numpy()
        
        return attributions_np, tokens, target_class, confidence
    
    @staticmethod
    def visualize(
        tokens: list,
        attributions: np.ndarray,
        predicted_class: int,
        confidence: float,
        class_names: Optional[list] = None,
        save_path: Optional[str] = None,
        show: bool = True,
        max_tokens: int = 50
    ) -> None:
        """Visualize Integrated Gradients attributions for text.
        
        Args:
            tokens: List of tokens
            attributions: Attribution scores for each token
            predicted_class: Predicted class index
            confidence: Prediction confidence
            class_names: List of class names
            save_path: Path to save figure
            show: Whether to display figure
            max_tokens: Maximum number of tokens to display
        """
        # Filter out special tokens and padding
        valid_indices = [
            i for i, t in enumerate(tokens[:max_tokens])
            if t not in ["[PAD]", "[CLS]", "[SEP]"]
        ]
        
        filtered_tokens = [tokens[i] for i in valid_indices]
        filtered_attrs = attributions[valid_indices]
        
        # Normalize attributions
        max_attr = max(abs(filtered_attrs.max()), abs(filtered_attrs.min()), 1e-8)
        norm_attrs = filtered_attrs / max_attr
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 4))
        
        # Create colored text visualization
        class_label = class_names[predicted_class] if class_names else str(predicted_class)
        
        # Use colormap for word highlighting
        cmap = plt.cm.get_cmap("RdYlGn")
        
        x_pos = 0.02
        y_pos = 0.7
        max_width = 0.96
        line_height = 0.15
        
        for i, (token, attr) in enumerate(zip(filtered_tokens, norm_attrs)):
            # Get color based on attribution
            color = cmap((attr + 1) / 2)  # Map [-1, 1] to [0, 1]
            
            # Handle wordpiece tokens
            display_token = token.replace("##", "")
            
            text_obj = ax.text(
                x_pos, y_pos, display_token + " ",
                fontsize=11,
                fontweight="bold",
                color="black",
                bbox=dict(boxstyle="round,pad=0.1", facecolor=color, edgecolor="none", alpha=0.7),
                transform=ax.transAxes
            )
            
            # Get text width
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bbox = text_obj.get_window_extent(renderer=renderer)
            text_width = bbox.width / (fig.get_figwidth() * fig.dpi)
            
            x_pos += text_width + 0.01
            
            # Wrap to next line
            if x_pos > max_width:
                x_pos = 0.02
                y_pos -= line_height
                if y_pos < 0.1:
                    break
        
        # Add title
        ax.set_title(f"Predicted: {class_label} (confidence: {confidence:.2%})", fontsize=12)
        ax.axis("off")
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-1, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation="horizontal", pad=0.02, fraction=0.05)
        cbar.set_label("Attribution Score (Negative ← → Positive)")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        
        if show:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def visualize_html(
        tokens: list,
        attributions: np.ndarray,
        predicted_class: int,
        confidence: float,
        class_names: Optional[list] = None,
        save_path: Optional[str] = None
    ) -> str:
        """Generate HTML visualization of text attributions.
        
        Args:
            tokens: List of tokens
            attributions: Attribution scores
            predicted_class: Predicted class index
            confidence: Prediction confidence
            class_names: List of class names
            save_path: Path to save HTML file
            
        Returns:
            HTML string
        """
        # Filter out special tokens and padding
        valid_indices = [
            i for i, t in enumerate(tokens)
            if t not in ["[PAD]", "[CLS]", "[SEP]"]
        ]
        
        filtered_tokens = [tokens[i] for i in valid_indices]
        filtered_attrs = attributions[valid_indices]
        
        # Normalize attributions
        max_attr = max(abs(filtered_attrs.max()), abs(filtered_attrs.min()), 1e-8)
        norm_attrs = filtered_attrs / max_attr
        
        class_label = class_names[predicted_class] if class_names else str(predicted_class)
        
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; padding: 20px; }}
                .title {{ font-size: 18px; margin-bottom: 20px; }}
                .text-container {{ line-height: 2.5; }}
                .token {{ padding: 2px 4px; margin: 1px; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="title">
                <strong>Predicted:</strong> {class_label} 
                (<strong>Confidence:</strong> {confidence:.2%})
            </div>
            <div class="text-container">
        """
        
        for token, attr in zip(filtered_tokens, norm_attrs):
            # Get color based on attribution
            if attr > 0:
                intensity = int(attr * 255)
                color = f"rgb({255 - intensity}, 255, {255 - intensity})"  # Green
            else:
                intensity = int(-attr * 255)
                color = f"rgb(255, {255 - intensity}, {255 - intensity})"  # Red
            
            display_token = token.replace("##", "")
            html += f'<span class="token" style="background-color: {color};">{display_token}</span> '
        
        html += """
            </div>
        </body>
        </html>
        """
        
        if save_path:
            with open(save_path, "w") as f:
                f.write(html)
        
        return html
