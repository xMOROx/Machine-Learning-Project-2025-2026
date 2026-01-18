"""Comprehensive Visualization Module for XAI Comparison.

This module provides rich visualization capabilities for comparing
DiET with GradCAM (images) and Integrated Gradients (text).

Features:
- Metric comparison bar charts
- Attribution heatmap overlays
- Side-by-side method comparisons
- Radar charts for multi-metric comparison
- Interactive HTML reports
- Notebook-friendly display functions
"""

import os
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as path_effects

try:
    import seaborn as sns

    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


@dataclass
class VisualizationConfig:
    """Configuration for visualizations."""

    figsize: Tuple[int, int] = (14, 8)
    dpi: int = 150
    colormap: str = "RdYlBu_r"
    style: str = "whitegrid"
    font_size: int = 12
    title_size: int = 14
    save_format: str = "png"
    # Enhanced styling
    use_gradients: bool = True
    color_palette: str = "husl"
    edge_color: str = "#333333"
    grid_alpha: float = 0.3
    bar_alpha: float = 0.85


MODERN_COLORS = {
    "gradcam": "#3498db",      # Bright blue
    "diet": "#2ecc71",          # Emerald green
    "ig": "#e74c3c",            # Vibrant red
    "baseline": "#95a5a6",      # Gray
    "primary": "#9b59b6",       # Purple
    "secondary": "#f39c12",     # Orange
    "accent": "#1abc9c",        # Teal
}

GRADIENT_COLORS = {
    "gradcam": ["#74b9ff", "#0984e3"],
    "diet": ["#55efc4", "#00b894"],
    "ig": ["#fab1a0", "#e17055"],
    "baseline": ["#dfe6e9", "#b2bec3"],
}


class ComparisonVisualizer:
    """Main visualization class for XAI method comparison.

    This class provides comprehensive visualization tools for comparing
    attribution methods like DiET, GradCAM, and Integrated Gradients.
    """

    def __init__(
        self,
        output_dir: str = "./outputs/visualizations",
        config: Optional[VisualizationConfig] = None,
    ):
        """Initialize ComparisonVisualizer.

        Args:
            output_dir: Directory to save visualizations
            config: Visualization configuration
        """
        self.output_dir = output_dir
        self.config = config or VisualizationConfig()
        os.makedirs(output_dir, exist_ok=True)

        if SEABORN_AVAILABLE:
            sns.set_style(self.config.style)
            sns.set_palette(self.config.color_palette)

        plt.rcParams.update(
            {
                "font.size": self.config.font_size,
                "axes.titlesize": self.config.title_size,
                "figure.dpi": self.config.dpi,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "font.family": "sans-serif",
                "axes.labelweight": "bold",
                "axes.titleweight": "bold",
            }
        )

        # Color schemes for methods
        self.method_colors = MODERN_COLORS

    def _add_value_labels(self, ax, bars, fmt=".3f", fontsize=9, offset=3):
        """Add value labels on top of bars."""
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:{fmt}}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=fontsize,
                fontweight="bold",
                color="#333333",
            )

    def _style_axis(self, ax, title="", xlabel="", ylabel=""):
        """Apply consistent styling to axis."""
        if title:
            ax.set_title(title, fontsize=self.config.title_size, fontweight="bold", pad=15)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=self.config.font_size, fontweight="bold")
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=self.config.font_size, fontweight="bold")
        ax.grid(True, alpha=self.config.grid_alpha, linestyle="--")
        ax.set_axisbelow(True)

    def plot_metric_comparison_bar(
        self,
        results: Dict[str, Dict[str, float]],
        title: str = "XAI Method Comparison",
        save_name: Optional[str] = None,
        show: bool = False,
    ) -> plt.Figure:
        """Create bar chart comparing metrics across methods.

        Args:
            results: Dictionary mapping method names to metric dictionaries
            title: Plot title
            save_name: Filename to save (without extension)
            show: Whether to display the plot

        Returns:
            matplotlib Figure object
        """
        methods = list(results.keys())
        metrics = list(results[methods[0]].keys())

        metrics = [m for m in metrics if not m.endswith("_std")]

        x = np.arange(len(metrics))
        width = 0.8 / len(methods)

        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        ax.set_facecolor("#fafafa")

        for i, method in enumerate(methods):
            values = [results[method].get(m, 0) for m in metrics]
            offset = (i - len(methods) / 2 + 0.5) * width

            color = self.method_colors.get(method.lower(), f"C{i}")
            bars = ax.bar(
                x + offset, 
                values, 
                width, 
                label=method, 
                color=color, 
                alpha=self.config.bar_alpha,
                edgecolor=self.config.edge_color,
                linewidth=1.5,
                zorder=3,
            )
            self._add_value_labels(ax, bars)

        self._style_axis(ax, title=title, xlabel="Metric", ylabel="Score")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha="right", fontsize=11)
        ax.legend(loc="upper right", framealpha=0.95, edgecolor="#cccccc")
        ax.set_ylim(0, max([max(results[m].get(metric, 0) for metric in metrics) for m in methods]) * 1.15)

        plt.tight_layout()

        if save_name:
            save_path = os.path.join(
                self.output_dir, f"{save_name}.{self.config.save_format}"
            )
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight", facecolor="white")
            print(f"Saved: {save_path}")

        if show:
            plt.show()

        return fig

    def plot_multi_metric_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        title: str = "Comprehensive Metric Comparison",
        save_name: Optional[str] = None,
        show: bool = False,
    ) -> plt.Figure:
        """Create grouped bar chart for multiple metrics.

        Args:
            results: Dictionary mapping method names to metric dictionaries
            title: Plot title
            save_name: Filename to save
            show: Whether to display

        Returns:
            matplotlib Figure object
        """
        methods = list(results.keys())
        all_metrics = set()
        for method_results in results.values():
            all_metrics.update(k for k in method_results.keys() if not k.endswith("_std"))
        metrics = sorted(list(all_metrics))
        
        n_metrics = len(metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = np.atleast_2d(axes)
        
        for idx, metric in enumerate(metrics):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]
            
            values = [results[m].get(metric, 0) for m in methods]
            stds = [results[m].get(f"{metric}_std", 0) for m in methods]
            colors = [self.method_colors.get(m.lower(), "#666666") for m in methods]
            
            bars = ax.bar(
                methods, 
                values, 
                color=colors, 
                alpha=self.config.bar_alpha,
                edgecolor=self.config.edge_color,
                linewidth=1.5,
                yerr=stds if any(s > 0 for s in stds) else None,
                capsize=5,
            )
            
            self._add_value_labels(ax, bars)
            self._style_axis(ax, title=metric.replace("_", " ").title())
            ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1)
        
        # Hide empty subplots
        for idx in range(n_metrics, n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            axes[row, col].axis("off")
        
        fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()
        
        if save_name:
            save_path = os.path.join(self.output_dir, f"{save_name}.{self.config.save_format}")
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight", facecolor="white")
            print(f"Saved: {save_path}")
        
        if show:
            plt.show()
        
        return fig

    def plot_radar_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        title: str = "Multi-Metric Comparison",
        save_name: Optional[str] = None,
        show: bool = False,
    ) -> plt.Figure:
        """Create radar/spider chart for multi-metric comparison.

        Args:
            results: Dictionary mapping method names to metric dictionaries
            title: Plot title
            save_name: Filename to save
            show: Whether to display

        Returns:
            matplotlib Figure object
        """
        methods = list(results.keys())
        metrics = [m for m in results[methods[0]].keys() if not m.endswith("_std")]

        normalized = {}
        for method in methods:
            normalized[method] = []
            for metric in metrics:
                values = [results[m].get(metric, 0) for m in methods]
                min_val, max_val = min(values), max(values)
                if max_val - min_val > 0:
                    norm_val = (results[method].get(metric, 0) - min_val) / (
                        max_val - min_val
                    )
                else:
                    norm_val = 0.5
                normalized[method].append(norm_val)

        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))
        ax.set_facecolor("#fafafa")

        for method in methods:
            values = normalized[method] + [normalized[method][0]]
            color = self.method_colors.get(method.lower(), None)
            ax.plot(angles, values, "o-", linewidth=2.5, label=method, color=color, markersize=8)
            ax.fill(angles, values, alpha=0.25, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace("_", "\n") for m in metrics], size=10, fontweight="bold")
        ax.set_title(title, size=self.config.title_size, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), framealpha=0.95)
        
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.xaxis.grid(True, linestyle="-", alpha=0.3)

        plt.tight_layout()

        if save_name:
            save_path = os.path.join(
                self.output_dir, f"{save_name}.{self.config.save_format}"
            )
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight", facecolor="white")
            print(f"Saved: {save_path}")

        if show:
            plt.show()

        return fig

    def plot_top_k_overlap_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        title: str = "Token Overlap Across K Values",
        save_name: Optional[str] = None,
        show: bool = False,
    ) -> plt.Figure:
        """Create line plot comparing top-k overlap across different k values.

        Args:
            results: Dictionary mapping dataset names to overlap metrics
            title: Plot title
            save_name: Filename to save
            show: Whether to display

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_facecolor("#fafafa")
        
        # Extract k values and overlaps for each dataset
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(results)))
        
        for idx, (dataset_name, metrics) in enumerate(results.items()):
            k_values = []
            overlaps = []
            stds = []
            
            for key, value in metrics.items():
                if key.startswith("top_") and key.endswith("_overlap") and not key.endswith("_std"):
                    k = int(key.split("_")[1])
                    k_values.append(k)
                    overlaps.append(value)
                    stds.append(metrics.get(f"{key}_std", 0))
            
            if k_values:
                sorted_idx = np.argsort(k_values)
                k_values = [k_values[i] for i in sorted_idx]
                overlaps = [overlaps[i] for i in sorted_idx]
                stds = [stds[i] for i in sorted_idx]
                
                ax.plot(
                    k_values, overlaps, 
                    "o-", 
                    linewidth=2.5, 
                    markersize=10,
                    label=dataset_name.upper(),
                    color=colors[idx],
                )
                ax.fill_between(
                    k_values,
                    [o - s for o, s in zip(overlaps, stds)],
                    [o + s for o, s in zip(overlaps, stds)],
                    alpha=0.2,
                    color=colors[idx],
                )
        
        self._style_axis(ax, title=title, xlabel="K (Top-K Tokens)", ylabel="Overlap Score")
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.5, color="#e74c3c", linestyle="--", alpha=0.5, label="50% Threshold")
        ax.legend(loc="best", framealpha=0.95)
        
        plt.tight_layout()
        
        if save_name:
            save_path = os.path.join(self.output_dir, f"{save_name}.{self.config.save_format}")
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight", facecolor="white")
            print(f"Saved: {save_path}")
        
        if show:
            plt.show()
        
        return fig

    def plot_image_attribution_comparison(
        self,
        image: np.ndarray,
        attributions: Dict[str, np.ndarray],
        true_label: str = "",
        predicted_label: str = "",
        confidence: float = 0.0,
        save_name: Optional[str] = None,
        show: bool = False,
    ) -> plt.Figure:
        """Create side-by-side comparison of image attributions.

        Args:
            image: Original image (H, W, 3)
            attributions: Dictionary mapping method names to attribution maps
            true_label: True class label
            predicted_label: Predicted class label
            confidence: Prediction confidence
            save_name: Filename to save
            show: Whether to display

        Returns:
            matplotlib Figure object
        """
        n_methods = len(attributions) + 1  # +1 for original image

        fig, axes = plt.subplots(1, n_methods + 1, figsize=(4 * (n_methods + 1), 5))
        fig.patch.set_facecolor("white")

        # Original image
        axes[0].imshow(image)
        axes[0].set_title(f"Original\nTrue: {true_label}", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        # Attribution overlays
        for i, (method_name, attr_map) in enumerate(attributions.items()):
            ax = axes[i + 1]

            # Resize attribution to image size
            if CV2_AVAILABLE:
                attr_resized = cv2.resize(attr_map, (image.shape[1], image.shape[0]))
            else:
                from scipy.ndimage import zoom

                zoom_factors = (
                    image.shape[0] / attr_map.shape[0],
                    image.shape[1] / attr_map.shape[1],
                )
                attr_resized = zoom(attr_map, zoom_factors, order=1)

            # Display image with overlay
            ax.imshow(image)
            im = ax.imshow(attr_resized, cmap="jet", alpha=0.5, vmin=0, vmax=1)
            ax.set_title(f"{method_name}", fontsize=11, fontweight="bold")
            ax.axis("off")

        # Difference map (if two methods)
        if len(attributions) == 2:
            methods = list(attributions.keys())
            attr1 = attributions[methods[0]]
            attr2 = attributions[methods[1]]

            # Resize both
            if CV2_AVAILABLE:
                attr1_resized = cv2.resize(attr1, (image.shape[1], image.shape[0]))
                attr2_resized = cv2.resize(attr2, (image.shape[1], image.shape[0]))
            else:
                from scipy.ndimage import zoom

                zoom_factors = (
                    image.shape[0] / attr1.shape[0],
                    image.shape[1] / attr1.shape[1],
                )
                attr1_resized = zoom(attr1, zoom_factors, order=1)
                attr2_resized = zoom(attr2, zoom_factors, order=1)

            diff = attr2_resized - attr1_resized
            axes[-1].imshow(diff, cmap="RdBu", vmin=-1, vmax=1)
            axes[-1].set_title(f"Difference\n{methods[1]} - {methods[0]}", fontsize=11, fontweight="bold")
            axes[-1].axis("off")

        # Add colorbar
        plt.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)

        fig.suptitle(
            f"Predicted: {predicted_label} (conf: {confidence:.2%})",
            fontsize=self.config.title_size,
            fontweight="bold",
        )

        plt.tight_layout()

        if save_name:
            save_path = os.path.join(
                self.output_dir, f"{save_name}.{self.config.save_format}"
            )
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight", facecolor="white")
            print(f"Saved: {save_path}")

        if show:
            plt.show()

        return fig

    def plot_text_attribution_comparison(
        self,
        tokens: List[str],
        attributions: Dict[str, np.ndarray],
        true_label: str = "",
        predicted_label: str = "",
        confidence: float = 0.0,
        max_tokens: int = 30,
        save_name: Optional[str] = None,
        show: bool = False,
    ) -> plt.Figure:
        """Create comparison of text attributions from different methods.

        Args:
            tokens: List of tokens
            attributions: Dictionary mapping method names to token attribution arrays
            true_label: True class label
            predicted_label: Predicted class label
            confidence: Prediction confidence
            max_tokens: Maximum tokens to display
            save_name: Filename to save
            show: Whether to display

        Returns:
            matplotlib Figure object
        """
        n_methods = len(attributions)

        special_tokens = {"[PAD]", "[CLS]", "[SEP]", "[UNK]", "<pad>", "<s>", "</s>"}
        valid_indices = [i for i, t in enumerate(tokens) if t not in special_tokens][:max_tokens]

        filtered_tokens = [tokens[i] for i in valid_indices]

        fig, axes = plt.subplots(n_methods, 1, figsize=(16, 3 * n_methods))
        if n_methods == 1:
            axes = [axes]
        
        fig.patch.set_facecolor("white")

        for ax, (method_name, attrs) in zip(axes, attributions.items()):
            # Filter attributions
            filtered_attrs = np.array(
                [attrs[i] for i in valid_indices if i < len(attrs)]
            )

            # Normalize
            if len(filtered_attrs) > 0:
                max_abs = max(np.abs(filtered_attrs).max(), 1e-8)
                norm_attrs = filtered_attrs / max_abs
            else:
                norm_attrs = filtered_attrs

            ax.set_xlim(-0.5, len(filtered_tokens) + 0.5)
            ax.set_ylim(-0.5, 0.5)
            
            cmap = plt.cm.RdYlGn
            
            for i, (token, attr) in enumerate(zip(filtered_tokens, norm_attrs)):
                color = cmap((attr + 1) / 2)
                display_token = token.replace("##", "")
                
                rect = FancyBboxPatch(
                    (i - 0.4, -0.3), 0.8, 0.6,
                    boxstyle="round,pad=0.05,rounding_size=0.1",
                    facecolor=color,
                    edgecolor="#333333",
                    linewidth=0.5,
                )
                ax.add_patch(rect)
                
                text_color = "white" if abs(attr) > 0.5 else "black"
                text = ax.text(
                    i, 0, display_token,
                    ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color=text_color,
                )
                text.set_path_effects([
                    path_effects.withStroke(linewidth=0.5, foreground="white" if text_color == "black" else "black")
                ])

            ax.set_title(f"{method_name}", fontsize=12, fontweight="bold", pad=10)
            ax.axis("off")

        fig.suptitle(
            f"Text Attribution Comparison\nTrue: {true_label} | "
            f"Predicted: {predicted_label} (conf: {confidence:.2%})",
            fontsize=self.config.title_size,
            fontweight="bold",
        )

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-1, 1))
        sm.set_array([])
        cbar = plt.colorbar(
            sm, ax=axes, orientation="horizontal", fraction=0.05, pad=0.15, aspect=50
        )
        cbar.set_label("Attribution Score (Negative ← → Positive)", fontweight="bold")

        plt.tight_layout()

        if save_name:
            save_path = os.path.join(
                self.output_dir, f"{save_name}.{self.config.save_format}"
            )
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight", facecolor="white")
            print(f"Saved: {save_path}")

        if show:
            plt.show()

        return fig

    def plot_perturbation_curve(
        self,
        results: Dict[str, Dict[int, float]],
        metric_type: str = "pixel_perturbation",
        title: str = "Perturbation Analysis",
        save_name: Optional[str] = None,
        show: bool = False,
    ) -> plt.Figure:
        """Plot perturbation curves for different methods.

        Args:
            results: Dictionary mapping method names to percentage->accuracy dicts
            metric_type: Type of perturbation metric
            title: Plot title
            save_name: Filename to save
            show: Whether to display

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_facecolor("#fafafa")

        for method_name, curve_data in results.items():
            percentages = sorted(curve_data.keys())
            accuracies = [curve_data[p] for p in percentages]

            color = self.method_colors.get(method_name.lower())
            ax.plot(
                percentages,
                accuracies,
                "o-",
                linewidth=2.5,
                label=method_name,
                color=color,
                markersize=10,
            )
            ax.fill_between(percentages, accuracies, alpha=0.15, color=color)

        self._style_axis(ax, title=title, xlabel="Percentage of Pixels", ylabel="Accuracy")
        ax.legend(loc="best", framealpha=0.95)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1.05)

        plt.tight_layout()

        if save_name:
            save_path = os.path.join(
                self.output_dir, f"{save_name}.{self.config.save_format}"
            )
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight", facecolor="white")
            print(f"Saved: {save_path}")

        if show:
            plt.show()

        return fig

    def plot_insertion_deletion_curves(
        self,
        insertion_curves: Dict[str, List[float]],
        deletion_curves: Dict[str, List[float]],
        title: str = "Insertion/Deletion Analysis",
        save_name: Optional[str] = None,
        show: bool = False,
    ) -> plt.Figure:
        """Plot insertion and deletion curves.

        Args:
            insertion_curves: Method name -> list of probabilities during insertion
            deletion_curves: Method name -> list of probabilities during deletion
            title: Plot title
            save_name: Filename to save
            show: Whether to display

        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for ax in axes:
            ax.set_facecolor("#fafafa")

        for method_name, probs in insertion_curves.items():
            x = np.linspace(0, 100, len(probs))
            color = self.method_colors.get(method_name.lower())
            axes[0].plot(x, probs, linewidth=2.5, label=method_name, color=color)
            axes[0].fill_between(x, probs, alpha=0.2, color=color)

        self._style_axis(axes[0], title="Insertion Curve\n(Higher is Better)", 
                        xlabel="% Pixels Inserted", ylabel="Probability")
        axes[0].legend(loc="lower right", framealpha=0.95)

        # Deletion curves
        for method_name, probs in deletion_curves.items():
            x = np.linspace(0, 100, len(probs))
            color = self.method_colors.get(method_name.lower())
            axes[1].plot(x, probs, linewidth=2.5, label=method_name, color=color)
            axes[1].fill_between(x, probs, alpha=0.2, color=color)

        self._style_axis(axes[1], title="Deletion Curve\n(Lower is Better)",
                        xlabel="% Pixels Deleted", ylabel="Probability")
        axes[1].legend(loc="upper right", framealpha=0.95)

        fig.suptitle(title, fontsize=self.config.title_size + 2, fontweight="bold")

        plt.tight_layout()

        if save_name:
            save_path = os.path.join(
                self.output_dir, f"{save_name}.{self.config.save_format}"
            )
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight", facecolor="white")
            print(f"Saved: {save_path}")

        if show:
            plt.show()

        return fig

    def create_summary_dashboard(
        self,
        image_results: Optional[Dict[str, Any]] = None,
        text_results: Optional[Dict[str, Any]] = None,
        save_name: str = "comparison_dashboard",
        show: bool = False,
    ) -> plt.Figure:
        """Create a comprehensive summary dashboard.

        Args:
            image_results: Results from image comparison (can be dict of datasets)
            text_results: Results from text comparison (can be dict of datasets)
            save_name: Filename to save
            show: Whether to display

        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=(18, 14))
        fig.patch.set_facecolor("white")
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

        # Title
        fig.suptitle(
            "DiET vs Basic XAI Methods - Comprehensive Comparison",
            fontsize=18,
            fontweight="bold",
            y=0.98,
        )

        if image_results:
            if isinstance(list(image_results.values())[0], dict) and "dataset" in list(image_results.values())[0]:
                datasets = list(image_results.keys())
                gradcam_scores = [image_results[ds].get("gradcam_mean_score", 0) for ds in datasets]
                diet_scores = [image_results[ds].get("diet_mean_score", 0) for ds in datasets]
            else:
                datasets = ["CIFAR-10"]
                gradcam_scores = [image_results.get("gradcam_mean_score", 0.5)]
                diet_scores = [image_results.get("diet_mean_score", 0.6)]

            ax1 = fig.add_subplot(gs[0, :2])
            ax1.set_facecolor("#fafafa")
            
            x = np.arange(len(datasets))
            width = 0.35

            bars1 = ax1.bar(
                x - width / 2,
                gradcam_scores,
                width,
                label="GradCAM",
                color=self.method_colors["gradcam"],
                alpha=self.config.bar_alpha,
                edgecolor=self.config.edge_color,
                linewidth=1.5,
            )
            bars2 = ax1.bar(
                x + width / 2,
                diet_scores,
                width,
                label="DiET",
                color=self.method_colors["diet"],
                alpha=self.config.bar_alpha,
                edgecolor=self.config.edge_color,
                linewidth=1.5,
            )

            self._add_value_labels(ax1, bars1, fmt=".3f")
            self._add_value_labels(ax1, bars2, fmt=".3f")
            self._style_axis(ax1, title="Image Attribution Quality (Higher = Better)", ylabel="Score")
            ax1.set_xticks(x)
            ax1.set_xticklabels([ds.upper() for ds in datasets], fontweight="bold")
            ax1.legend(framealpha=0.95)
            ax1.set_ylim(0, max(max(gradcam_scores), max(diet_scores)) * 1.2)

            ax2 = fig.add_subplot(gs[0, 2])
            ax2.set_facecolor("#fafafa")
            
            if isinstance(list(image_results.values())[0], dict) and "improvement" in list(image_results.values())[0]:
                improvements = [image_results[ds].get("improvement", 0) for ds in datasets]
            else:
                improvements = [image_results.get("improvement", 0)]
            
            colors = [self.method_colors["diet"] if imp > 0 else self.method_colors["gradcam"] for imp in improvements]
            bars = ax2.barh(
                [ds.upper() for ds in datasets], 
                improvements, 
                color=colors, 
                alpha=self.config.bar_alpha,
                edgecolor=self.config.edge_color,
            )
            ax2.axvline(x=0, color="black", linestyle="-", linewidth=1)
            self._style_axis(ax2, title="DiET Improvement", xlabel="Δ Score")
            for bar, val in zip(bars, improvements):
                ax2.text(
                    val + 0.01 if val >= 0 else val - 0.01, 
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:+.3f}", 
                    va="center", 
                    ha="left" if val >= 0 else "right",
                    fontsize=10, 
                    fontweight="bold"
                )

        if text_results:
            ax3 = fig.add_subplot(gs[1, :2])
            ax3.set_facecolor("#fafafa")

            if isinstance(list(text_results.values())[0], dict) and "dataset" in list(text_results.values())[0]:
                datasets = list(text_results.keys())
            else:
                datasets = ["SST-2"]
            
            k_values = [3, 5, 10, 15, 20]
            x = np.arange(len(datasets))
            width = 0.15
            
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(k_values)))
            
            for i, k in enumerate(k_values):
                key = f"top_{k}_overlap"
                if isinstance(list(text_results.values())[0], dict):
                    overlaps = [text_results[ds].get(key, text_results[ds].get("ig_diet_overlap", 0)) for ds in datasets]
                else:
                    overlaps = [text_results.get(key, text_results.get("ig_diet_overlap", 0.4))]
                
                offset = (i - len(k_values) / 2 + 0.5) * width
                bars = ax3.bar(x + offset, overlaps, width, label=f"Top-{k}", color=colors[i], alpha=0.8)

            self._style_axis(ax3, title="Text Attribution: IG-DiET Token Overlap", ylabel="Overlap Score")
            ax3.set_xticks(x)
            ax3.set_xticklabels([ds.upper() for ds in datasets], fontweight="bold")
            ax3.axhline(y=0.5, color="#e74c3c", linestyle="--", alpha=0.7, label="50% Threshold")
            ax3.legend(loc="upper right", ncol=3, framealpha=0.95)
            ax3.set_ylim(0, 1.1)

            ax4 = fig.add_subplot(gs[1, 2])
            ax4.set_facecolor("#fafafa")
            
            if isinstance(list(text_results.values())[0], dict):
                correlations = [text_results[ds].get("mean_correlation", 0) for ds in datasets]
            else:
                correlations = [text_results.get("mean_correlation", 0)]
            
            bars = ax4.bar(
                [ds.upper() for ds in datasets], 
                correlations, 
                color=self.method_colors["ig"],
                alpha=self.config.bar_alpha,
                edgecolor=self.config.edge_color,
            )
            self._add_value_labels(ax4, bars, fmt=".3f")
            self._style_axis(ax4, title="Attribution Correlation", ylabel="Correlation")
            ax4.set_ylim(-1, 1)
            ax4.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

        # Summary panel
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis("off")

        summary_text = "📊 EXPERIMENT SUMMARY\n" + "=" * 60 + "\n\n"

        if image_results:
            if isinstance(list(image_results.values())[0], dict):
                valid_results = [v for v in image_results.values() if "error" not in v]
                diet_wins = sum(1 for v in valid_results if v.get("diet_better", False))
                avg_imp = np.mean([v.get("improvement", 0) for v in valid_results])
                summary_text += f"🖼️ IMAGE EXPERIMENTS (DiET vs GradCAM):\n"
                summary_text += f"   • DiET outperforms GradCAM: {diet_wins}/{len(valid_results)} datasets\n"
                summary_text += f"   • Average improvement: {avg_imp:+.4f}\n\n"
            else:
                improvement = image_results.get("improvement", 0)
                summary_text += f"🖼️ IMAGE EXPERIMENTS:\n"
                summary_text += f"   • Improvement: {improvement:+.4f}\n\n"

        if text_results:
            if isinstance(list(text_results.values())[0], dict):
                valid_results = [v for v in text_results.values() if "error" not in v]
                avg_overlap = np.mean([v.get("ig_diet_overlap", v.get("top_5_overlap", 0)) for v in valid_results])
                avg_corr = np.mean([v.get("mean_correlation", 0) for v in valid_results])
                summary_text += f"📝 TEXT EXPERIMENTS (DiET vs Integrated Gradients):\n"
                summary_text += f"   • Average token overlap: {avg_overlap:.4f}\n"
                summary_text += f"   • Average correlation: {avg_corr:.4f}\n"
            else:
                overlap = text_results.get("ig_diet_overlap", 0)
                summary_text += f"📝 TEXT EXPERIMENTS:\n"
                summary_text += f"   • IG-DiET overlap: {overlap:.4f}\n"

        ax5.text(
            0.5,
            0.5,
            summary_text,
            transform=ax5.transAxes,
            fontsize=12,
            verticalalignment="center",
            horizontalalignment="center",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#e8f4f8", edgecolor="#3498db", alpha=0.9),
        )

        if save_name:
            save_path = os.path.join(
                self.output_dir, f"{save_name}.{self.config.save_format}"
            )
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight", facecolor="white")
            print(f"Saved: {save_path}")

        if show:
            plt.show()

        return fig

    def generate_html_report(
        self,
        results: Dict[str, Any],
        report_title: str = "XAI Comparison Report",
        save_name: str = "comparison_report",
    ) -> str:
        """Generate an interactive HTML report.

        Args:
            results: All comparison results
            report_title: Title of the report
            save_name: Filename to save (without extension)

        Returns:
            Path to saved HTML file
        """
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report_title}</title>
    <style>
        :root {{
            --primary-color: #3498db;
            --secondary-color: #2ecc71;
            --accent-color: #e74c3c;
            --bg-color: #f8f9fa;
            --card-bg: #ffffff;
            --text-color: #2c3e50;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: var(--text-color);
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: var(--bg-color);
        }}
        h1 {{
            color: var(--primary-color);
            border-bottom: 3px solid var(--primary-color);
            padding-bottom: 15px;
            font-size: 2.2em;
        }}
        h2 {{
            color: var(--secondary-color);
            margin-top: 40px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .section {{
            background: var(--card-bg);
            padding: 25px;
            border-radius: 12px;
            margin-bottom: 25px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-left: 4px solid var(--primary-color);
        }}
        .metric-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            border-radius: 8px;
            overflow: hidden;
        }}
        .metric-table th, .metric-table td {{
            padding: 14px 18px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        .metric-table th {{
            background: linear-gradient(135deg, var(--secondary-color), #27ae60);
            color: white;
            font-weight: 600;
        }}
        .metric-table tr:hover {{
            background-color: #f0f8ff;
        }}
        .highlight {{
            background-color: #d4edda;
            font-weight: bold;
        }}
        .summary-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 12px;
            margin-top: 30px;
        }}
        .badge {{
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            margin-right: 8px;
        }}
        .badge-gradcam {{ background: linear-gradient(135deg, #74b9ff, #0984e3); color: white; }}
        .badge-diet {{ background: linear-gradient(135deg, #55efc4, #00b894); color: white; }}
        .badge-ig {{ background: linear-gradient(135deg, #fab1a0, #e17055); color: white; }}
        .metric-card {{
            display: inline-block;
            background: #f8f9fa;
            padding: 15px 20px;
            border-radius: 8px;
            margin: 5px;
            text-align: center;
            min-width: 120px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: var(--primary-color);
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }}
        .progress-bar {{
            background: #e0e0e0;
            border-radius: 10px;
            height: 20px;
            overflow: hidden;
        }}
        .progress-fill {{
            height: 100%;
            border-radius: 10px;
            transition: width 0.3s ease;
        }}
    </style>
</head>
<body>
    <h1>🔬 {report_title}</h1>
    <p style="color: #666;">Generated: {np.datetime64("now")}</p>
"""

        # Image section
        if "image_experiments" in results and results["image_experiments"]:
            html += """
    <div class="section">
        <h2>📸 Image Classification Results</h2>
        <p>Comparison of <span class="badge badge-gradcam">GradCAM</span> vs 
           <span class="badge badge-diet">DiET</span></p>
        <table class="metric-table">
            <tr>
                <th>Dataset</th>
                <th>GradCAM Score</th>
                <th>DiET Score</th>
                <th>Improvement</th>
                <th>Winner</th>
            </tr>
"""
            for dataset_name, img in results["image_experiments"].items():
                if "error" in img:
                    continue
                gradcam_score = img.get("gradcam_mean_score", 0)
                diet_score = img.get("diet_mean_score", 0)
                improvement = img.get("improvement", 0)
                winner = "DiET ✓" if improvement > 0 else "GradCAM"
                winner_class = "highlight" if improvement > 0 else ""
                
                html += f"""
            <tr>
                <td><strong>{dataset_name.upper()}</strong></td>
                <td>{gradcam_score:.4f}</td>
                <td>{diet_score:.4f}</td>
                <td style="color: {'green' if improvement > 0 else 'red'}">{improvement:+.4f}</td>
                <td class="{winner_class}">{winner}</td>
            </tr>
"""
            html += """
        </table>
    </div>
"""

        # Text section
        if "text_experiments" in results and results["text_experiments"]:
            html += """
    <div class="section">
        <h2>📝 Text Classification Results</h2>
        <p>Comparison of <span class="badge badge-ig">Integrated Gradients</span> vs 
           <span class="badge badge-diet">DiET</span></p>
        <table class="metric-table">
            <tr>
                <th>Dataset</th>
                <th>Accuracy</th>
                <th>Top-3 Overlap</th>
                <th>Top-5 Overlap</th>
                <th>Top-10 Overlap</th>
                <th>Correlation</th>
            </tr>
"""
            for dataset_name, txt in results["text_experiments"].items():
                if "error" in txt:
                    continue
                html += f"""
            <tr>
                <td><strong>{dataset_name.upper()}</strong></td>
                <td>{txt.get('baseline_accuracy', 0):.1f}%</td>
                <td>{txt.get('top_3_overlap', txt.get('ig_diet_overlap', 0)):.3f}</td>
                <td>{txt.get('top_5_overlap', txt.get('ig_diet_overlap', 0)):.3f}</td>
                <td>{txt.get('top_10_overlap', txt.get('ig_diet_overlap', 0)):.3f}</td>
                <td>{txt.get('mean_correlation', 0):.3f}</td>
            </tr>
"""
            html += """
        </table>
    </div>
"""

        # Summary
        html += """
    <div class="summary-box">
        <h2 style="color: white; margin-top: 0;">📊 Key Findings</h2>
        <ul style="font-size: 16px;">
"""

        if "image_experiments" in results and results["image_experiments"]:
            valid = [v for v in results["image_experiments"].values() if "error" not in v]
            diet_wins = sum(1 for v in valid if v.get("diet_better", False))
            html += f"<li>✅ Image: DiET outperforms GradCAM on {diet_wins}/{len(valid)} datasets</li>"

        if "text_experiments" in results and results["text_experiments"]:
            valid = [v for v in results["text_experiments"].values() if "error" not in v]
            avg_overlap = np.mean([v.get("top_5_overlap", v.get("ig_diet_overlap", 0)) for v in valid])
            avg_corr = np.mean([v.get("mean_correlation", 0) for v in valid])
            html += f"<li>📝 Text: Average token overlap = {avg_overlap:.3f}, correlation = {avg_corr:.3f}</li>"

        html += """
        </ul>
    </div>
</body>
</html>
"""

        save_path = os.path.join(self.output_dir, f"{save_name}.html")
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"Saved HTML report: {save_path}")
        return save_path


def plot_metric_comparison(
    results: Dict[str, Dict[str, float]], output_dir: str = "./outputs", **kwargs
) -> plt.Figure:
    """Convenience function for quick metric comparison plot."""
    visualizer = ComparisonVisualizer(output_dir=output_dir)
    return visualizer.plot_metric_comparison_bar(results, **kwargs)


def plot_attribution_comparison(
    image: np.ndarray,
    attributions: Dict[str, np.ndarray],
    output_dir: str = "./outputs",
    **kwargs,
) -> plt.Figure:
    """Convenience function for quick attribution comparison plot."""
    visualizer = ComparisonVisualizer(output_dir=output_dir)
    return visualizer.plot_image_attribution_comparison(image, attributions, **kwargs)


def create_comparison_report(
    results: Dict[str, Any], output_dir: str = "./outputs", **kwargs
) -> str:
    """Convenience function for generating HTML report."""
    visualizer = ComparisonVisualizer(output_dir=output_dir)
    return visualizer.generate_html_report(results, **kwargs)
