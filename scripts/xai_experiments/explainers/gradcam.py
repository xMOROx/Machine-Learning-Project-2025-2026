"""GradCAM Implementation for CNN Explainability.

This module implements Gradient-weighted Class Activation Mapping (GradCAM)
for visualizing which parts of an image are important for classification.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization", ICCV 2017
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class GradCAM:
    """Gradient-weighted Class Activation Mapping for CNN visualization.
    
    GradCAM uses the gradients flowing into the final convolutional layer
    to produce a coarse localization map highlighting important regions
    in the image for predicting the target class.
    
    Attributes:
        model: The CNN model to explain
        device: Device to run computations on
    """
    
    def __init__(self, model: torch.nn.Module, device: str = "cuda"):
        """Initialize GradCAM.
        
        Args:
            model: CNN model with get_activations() and get_activations_gradient() methods
            device: Device to run computations on
        """
        self.model = model
        self.device = device
        self.model.eval()
    
    def generate_cam(
        self,
        input_image: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        """Generate GradCAM heatmap for an input image.
        
        Args:
            input_image: Input tensor of shape (1, C, H, W)
            target_class: Class to generate CAM for. If None, uses predicted class.
            
        Returns:
            Tuple of (heatmap, predicted_class, confidence)
        """
        input_image = input_image.to(self.device)
        input_image.requires_grad_(True)
        
        # Forward pass
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Get confidence (softmax probability)
        probs = F.softmax(output, dim=1)
        confidence = probs[0, target_class].item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.model.get_activations_gradient()
        activations = self.model.get_activations()
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1).squeeze()
        
        # ReLU to keep only positive contributions
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Convert to numpy
        cam = cam.detach().cpu().numpy()
        
        return cam, target_class, confidence
    
    def generate_cam_batch(
        self,
        images: torch.Tensor,
        target_classes: Optional[torch.Tensor] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate GradCAM heatmaps for a batch of images.
        
        Args:
            images: Input tensor of shape (B, C, H, W)
            target_classes: Classes to generate CAM for. If None, uses predicted classes.
            
        Returns:
            Tuple of (heatmaps, predicted_classes, confidences)
        """
        batch_size = images.shape[0]
        heatmaps = []
        predicted_classes = []
        confidences = []
        
        for i in range(batch_size):
            cam, pred_class, conf = self.generate_cam(
                images[i:i+1],
                target_classes[i].item() if target_classes is not None else None
            )
            heatmaps.append(cam)
            predicted_classes.append(pred_class)
            confidences.append(conf)
        
        return np.array(heatmaps), np.array(predicted_classes), np.array(confidences)
    
    @staticmethod
    def apply_colormap(heatmap: np.ndarray, colormap: str = "jet") -> np.ndarray:
        """Apply colormap to heatmap.
        
        Args:
            heatmap: 2D numpy array of shape (H, W)
            colormap: Matplotlib colormap name
            
        Returns:
            RGB image of shape (H, W, 3)
        """
        cmap = cm.get_cmap(colormap)
        colored = cmap(heatmap)[:, :, :3]  # Remove alpha channel
        return (colored * 255).astype(np.uint8)
    
    @staticmethod
    def overlay_heatmap(
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: str = "jet"
    ) -> np.ndarray:
        """Overlay heatmap on original image.
        
        Args:
            image: Original image of shape (H, W, 3), values in [0, 255]
            heatmap: Heatmap of shape (H', W')
            alpha: Blending factor for heatmap
            colormap: Matplotlib colormap name
            
        Returns:
            Overlaid image of shape (H, W, 3)
        """
        import cv2
        
        # Resize heatmap to match image size
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Apply colormap
        colored_heatmap = GradCAM.apply_colormap(heatmap_resized, colormap)
        
        # Blend
        if image.max() <= 1:
            image = (image * 255).astype(np.uint8)
        
        overlaid = cv2.addWeighted(image, 1 - alpha, colored_heatmap, alpha, 0)
        
        return overlaid
    
    @staticmethod
    def visualize(
        original_image: np.ndarray,
        heatmap: np.ndarray,
        predicted_class: int,
        confidence: float,
        class_names: Optional[list] = None,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """Visualize GradCAM results.
        
        Args:
            original_image: Original image of shape (H, W, 3)
            heatmap: GradCAM heatmap
            predicted_class: Predicted class index
            confidence: Prediction confidence
            class_names: List of class names for labeling
            save_path: Path to save the figure
            show: Whether to display the figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        
        # Heatmap
        import cv2
        heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        axes[1].imshow(heatmap_resized, cmap="jet")
        axes[1].set_title("GradCAM Heatmap")
        axes[1].axis("off")
        
        # Overlay
        overlaid = GradCAM.overlay_heatmap(original_image, heatmap)
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
    
    @staticmethod
    def visualize_batch(
        images: np.ndarray,
        heatmaps: np.ndarray,
        predicted_classes: np.ndarray,
        confidences: np.ndarray,
        class_names: Optional[list] = None,
        save_path: Optional[str] = None,
        max_images: int = 16,
        show: bool = True
    ) -> None:
        """Visualize GradCAM results for a batch of images.
        
        Args:
            images: Original images of shape (B, H, W, 3)
            heatmaps: GradCAM heatmaps of shape (B, H', W')
            predicted_classes: Predicted class indices
            confidences: Prediction confidences
            class_names: List of class names
            save_path: Path to save the figure
            max_images: Maximum number of images to display
            show: Whether to display the figure
        """
        import cv2
        
        n_images = min(len(images), max_images)
        n_cols = min(4, n_images)
        n_rows = (n_images + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_images):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col]
            
            # Create overlay
            img = images[i]
            if img.max() <= 1:
                img = (img * 255).astype(np.uint8)
            
            heatmap_resized = cv2.resize(heatmaps[i], (img.shape[1], img.shape[0]))
            overlaid = GradCAM.overlay_heatmap(img, heatmaps[i])
            
            ax.imshow(overlaid)
            class_label = class_names[predicted_classes[i]] if class_names else str(predicted_classes[i])
            ax.set_title(f"{class_label} ({confidences[i]:.2%})", fontsize=10)
            ax.axis("off")
        
        # Hide empty subplots
        for i in range(n_images, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].axis("off")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        
        if show:
            plt.show()
        
        plt.close()
