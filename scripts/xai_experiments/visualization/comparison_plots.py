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
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

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


class ComparisonVisualizer:
    """Main visualization class for XAI method comparison.
    
    This class provides comprehensive visualization tools for comparing
    attribution methods like DiET, GradCAM, and Integrated Gradients.
    """
    
    def __init__(
        self,
        output_dir: str = "./outputs/visualizations",
        config: Optional[VisualizationConfig] = None
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
        
        plt.rcParams.update({
            'font.size': self.config.font_size,
            'axes.titlesize': self.config.title_size,
            'figure.dpi': self.config.dpi
        })
        
        # Color schemes for methods
        self.method_colors = {
            "gradcam": "#2196F3",  # Blue
            "diet": "#4CAF50",     # Green
            "ig": "#FF9800",       # Orange
            "baseline": "#9E9E9E"  # Gray
        }
    
    def plot_metric_comparison_bar(
        self,
        results: Dict[str, Dict[str, float]],
        title: str = "XAI Method Comparison",
        save_name: Optional[str] = None,
        show: bool = False
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
        
        # Filter out std columns
        metrics = [m for m in metrics if not m.endswith("_std")]
        
        x = np.arange(len(metrics))
        width = 0.8 / len(methods)
        
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        for i, method in enumerate(methods):
            values = [results[method].get(m, 0) for m in metrics]
            offset = (i - len(methods) / 2 + 0.5) * width
            
            color = self.method_colors.get(method.lower(), f"C{i}")
            bars = ax.bar(x + offset, values, width, label=method, color=color, alpha=0.8)
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f'{val:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=8)
        
        ax.set_xlabel('Metric', fontsize=self.config.font_size)
        ax.set_ylabel('Score', fontsize=self.config.font_size)
        ax.set_title(title, fontsize=self.config.title_size, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            save_path = os.path.join(self.output_dir, f"{save_name}.{self.config.save_format}")
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_radar_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        title: str = "Multi-Metric Comparison",
        save_name: Optional[str] = None,
        show: bool = False
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
        
        # Normalize values to [0, 1] for radar chart
        normalized = {}
        for method in methods:
            normalized[method] = []
            for metric in metrics:
                values = [results[m].get(metric, 0) for m in methods]
                min_val, max_val = min(values), max(values)
                if max_val - min_val > 0:
                    norm_val = (results[method].get(metric, 0) - min_val) / (max_val - min_val)
                else:
                    norm_val = 0.5
                normalized[method].append(norm_val)
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        for method in methods:
            values = normalized[method] + [normalized[method][0]]
            color = self.method_colors.get(method.lower(), None)
            ax.plot(angles, values, 'o-', linewidth=2, label=method, color=color)
            ax.fill(angles, values, alpha=0.25, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, size=10)
        ax.set_title(title, size=self.config.title_size, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        
        if save_name:
            save_path = os.path.join(self.output_dir, f"{save_name}.{self.config.save_format}")
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
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
        show: bool = False
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
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title(f"Original\nTrue: {true_label}", fontsize=11)
        axes[0].axis('off')
        
        # Attribution overlays
        for i, (method_name, attr_map) in enumerate(attributions.items()):
            ax = axes[i + 1]
            
            # Resize attribution to image size
            if CV2_AVAILABLE:
                attr_resized = cv2.resize(attr_map, (image.shape[1], image.shape[0]))
            else:
                from scipy.ndimage import zoom
                zoom_factors = (image.shape[0] / attr_map.shape[0],
                               image.shape[1] / attr_map.shape[1])
                attr_resized = zoom(attr_map, zoom_factors, order=1)
            
            # Display image with overlay
            ax.imshow(image)
            im = ax.imshow(attr_resized, cmap='jet', alpha=0.5, vmin=0, vmax=1)
            ax.set_title(f"{method_name}", fontsize=11, fontweight='bold')
            ax.axis('off')
        
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
                zoom_factors = (image.shape[0] / attr1.shape[0],
                               image.shape[1] / attr1.shape[1])
                attr1_resized = zoom(attr1, zoom_factors, order=1)
                attr2_resized = zoom(attr2, zoom_factors, order=1)
            
            diff = attr2_resized - attr1_resized
            axes[-1].imshow(diff, cmap='RdBu', vmin=-1, vmax=1)
            axes[-1].set_title(f"Difference\n{methods[1]} - {methods[0]}", fontsize=11)
            axes[-1].axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)
        
        fig.suptitle(f"Predicted: {predicted_label} (conf: {confidence:.2%})",
                    fontsize=self.config.title_size, fontweight='bold')
        
        plt.tight_layout()
        
        if save_name:
            save_path = os.path.join(self.output_dir, f"{save_name}.{self.config.save_format}")
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
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
        save_name: Optional[str] = None,
        show: bool = False
    ) -> plt.Figure:
        """Create comparison of text attributions from different methods.
        
        Args:
            tokens: List of tokens
            attributions: Dictionary mapping method names to token attribution arrays
            true_label: True class label
            predicted_label: Predicted class label
            confidence: Prediction confidence
            save_name: Filename to save
            show: Whether to display
            
        Returns:
            matplotlib Figure object
        """
        n_methods = len(attributions)
        
        # Filter special tokens
        special_tokens = {'[PAD]', '[CLS]', '[SEP]', '[UNK]', '<pad>', '<s>', '</s>'}
        valid_indices = [i for i, t in enumerate(tokens) if t not in special_tokens]
        
        filtered_tokens = [tokens[i] for i in valid_indices]
        
        fig, axes = plt.subplots(n_methods, 1, figsize=(14, 3 * n_methods))
        if n_methods == 1:
            axes = [axes]
        
        for ax, (method_name, attrs) in zip(axes, attributions.items()):
            # Filter attributions
            filtered_attrs = np.array([attrs[i] for i in valid_indices if i < len(attrs)])
            
            # Normalize
            if len(filtered_attrs) > 0:
                max_abs = max(np.abs(filtered_attrs).max(), 1e-8)
                norm_attrs = filtered_attrs / max_abs
            else:
                norm_attrs = filtered_attrs
            
            # Create color bar for each token
            cmap = plt.cm.RdYlGn
            colors = [cmap((a + 1) / 2) for a in norm_attrs]
            
            # Display as horizontal bar
            x_positions = np.arange(len(filtered_tokens))
            ax.barh(0, np.ones(len(filtered_tokens)), color=colors, height=0.8)
            
            # Add token labels
            for i, (token, attr) in enumerate(zip(filtered_tokens, norm_attrs)):
                display_token = token.replace('##', '')
                text_color = 'white' if abs(attr) > 0.5 else 'black'
                ax.text(i, 0, display_token, ha='center', va='center',
                       fontsize=8, color=text_color, rotation=45)
            
            ax.set_xlim(-0.5, len(filtered_tokens) - 0.5)
            ax.set_ylim(-0.5, 0.5)
            ax.set_title(f"{method_name}", fontsize=11, fontweight='bold')
            ax.axis('off')
        
        fig.suptitle(f"Text Attribution Comparison\nTrue: {true_label} | "
                    f"Predicted: {predicted_label} (conf: {confidence:.2%})",
                    fontsize=self.config.title_size, fontweight='bold')
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-1, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=axes, orientation='horizontal', 
                           fraction=0.05, pad=0.1, aspect=50)
        cbar.set_label('Attribution Score (Negative ← → Positive)')
        
        plt.tight_layout()
        
        if save_name:
            save_path = os.path.join(self.output_dir, f"{save_name}.{self.config.save_format}")
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
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
        show: bool = False
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
        
        for method_name, curve_data in results.items():
            percentages = sorted(curve_data.keys())
            accuracies = [curve_data[p] for p in percentages]
            
            color = self.method_colors.get(method_name.lower())
            ax.plot(percentages, accuracies, 'o-', linewidth=2,
                   label=method_name, color=color, markersize=8)
        
        ax.set_xlabel('Percentage of Pixels', fontsize=self.config.font_size)
        ax.set_ylabel('Accuracy', fontsize=self.config.font_size)
        ax.set_title(title, fontsize=self.config.title_size, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1.05)
        
        plt.tight_layout()
        
        if save_name:
            save_path = os.path.join(self.output_dir, f"{save_name}.{self.config.save_format}")
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
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
        show: bool = False
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
        
        # Insertion curves
        for method_name, probs in insertion_curves.items():
            x = np.linspace(0, 100, len(probs))
            color = self.method_colors.get(method_name.lower())
            axes[0].plot(x, probs, linewidth=2, label=method_name, color=color)
            axes[0].fill_between(x, probs, alpha=0.2, color=color)
        
        axes[0].set_xlabel('% Pixels Inserted', fontsize=self.config.font_size)
        axes[0].set_ylabel('Probability', fontsize=self.config.font_size)
        axes[0].set_title('Insertion Curve\n(Higher is Better)', 
                         fontsize=self.config.title_size)
        axes[0].legend(loc='lower right')
        axes[0].grid(True, alpha=0.3)
        
        # Deletion curves
        for method_name, probs in deletion_curves.items():
            x = np.linspace(0, 100, len(probs))
            color = self.method_colors.get(method_name.lower())
            axes[1].plot(x, probs, linewidth=2, label=method_name, color=color)
            axes[1].fill_between(x, probs, alpha=0.2, color=color)
        
        axes[1].set_xlabel('% Pixels Deleted', fontsize=self.config.font_size)
        axes[1].set_ylabel('Probability', fontsize=self.config.font_size)
        axes[1].set_title('Deletion Curve\n(Lower is Better)', 
                         fontsize=self.config.title_size)
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=self.config.title_size + 2, fontweight='bold')
        
        plt.tight_layout()
        
        if save_name:
            save_path = os.path.join(self.output_dir, f"{save_name}.{self.config.save_format}")
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def create_summary_dashboard(
        self,
        image_results: Optional[Dict[str, Any]] = None,
        text_results: Optional[Dict[str, Any]] = None,
        save_name: str = "comparison_dashboard",
        show: bool = False
    ) -> plt.Figure:
        """Create a comprehensive summary dashboard.
        
        Args:
            image_results: Results from image comparison
            text_results: Results from text comparison
            save_name: Filename to save
            show: Whether to display
            
        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle("DiET vs Basic XAI Methods - Comprehensive Comparison",
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Image results section
        if image_results:
            # Metric comparison bar chart
            ax1 = fig.add_subplot(gs[0, :2])
            methods = ['GradCAM', 'DiET']
            metrics = ['Pixel Perturbation', 'AOPC', 'Faithfulness']
            
            x = np.arange(len(metrics))
            width = 0.35
            
            gradcam_scores = [
                image_results.get('gradcam_perturbation', 0.5),
                image_results.get('gradcam_aopc', 0.3),
                image_results.get('gradcam_faithfulness', 0.4)
            ]
            diet_scores = [
                image_results.get('diet_perturbation', 0.6),
                image_results.get('diet_aopc', 0.4),
                image_results.get('diet_faithfulness', 0.5)
            ]
            
            ax1.bar(x - width/2, gradcam_scores, width, label='GradCAM',
                   color=self.method_colors['gradcam'], alpha=0.8)
            ax1.bar(x + width/2, diet_scores, width, label='DiET',
                   color=self.method_colors['diet'], alpha=0.8)
            
            ax1.set_ylabel('Score')
            ax1.set_title('Image Attribution Metrics (CIFAR-10)', fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(metrics)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Accuracy comparison
            ax2 = fig.add_subplot(gs[0, 2])
            accs = [
                image_results.get('baseline_accuracy', 85),
                image_results.get('diet_accuracy', 84)
            ]
            bars = ax2.bar(['Baseline', 'After DiET'], accs,
                          color=[self.method_colors['baseline'], self.method_colors['diet']])
            ax2.set_ylabel('Accuracy (%)')
            ax2.set_title('Model Accuracy', fontweight='bold')
            ax2.set_ylim(0, 100)
            for bar, acc in zip(bars, accs):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{acc:.1f}%', ha='center', fontsize=10)
        
        # Text results section
        if text_results:
            ax3 = fig.add_subplot(gs[1, :2])
            
            # Token overlap comparison
            overlaps = [
                text_results.get('top_3_overlap', 0.4),
                text_results.get('top_5_overlap', 0.35),
                text_results.get('top_10_overlap', 0.3)
            ]
            labels = ['Top-3', 'Top-5', 'Top-10']
            
            colors = [self.method_colors['ig'], 
                     self.method_colors['diet'],
                     self.method_colors['baseline']]
            
            ax3.bar(labels, overlaps, color=self.method_colors['ig'], alpha=0.8)
            ax3.set_ylabel('Overlap Score')
            ax3.set_title('Text Attribution: IG vs DiET Token Overlap (SST-2)', fontweight='bold')
            ax3.set_ylim(0, 1)
            ax3.grid(True, alpha=0.3)
            
            for i, (label, overlap) in enumerate(zip(labels, overlaps)):
                ax3.text(i, overlap + 0.02, f'{overlap:.2f}', ha='center', fontsize=10)
            
            # Text accuracy
            ax4 = fig.add_subplot(gs[1, 2])
            text_acc = text_results.get('baseline_accuracy', 88)
            ax4.bar(['BERT Baseline'], [text_acc], 
                   color=self.method_colors['baseline'], alpha=0.8)
            ax4.set_ylabel('Accuracy (%)')
            ax4.set_title('Text Model Accuracy', fontweight='bold')
            ax4.set_ylim(0, 100)
            ax4.text(0, text_acc + 1, f'{text_acc:.1f}%', ha='center', fontsize=10)
        
        # Summary panel
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        summary_text = "Summary:\n"
        if image_results:
            improvement = image_results.get('improvement', 0)
            if improvement > 0:
                summary_text += f"• Image: DiET improves attribution quality by {improvement:.4f}\n"
            else:
                summary_text += f"• Image: GradCAM performs better by {-improvement:.4f}\n"
        
        if text_results:
            overlap = text_results.get('mean_overlap', 0)
            summary_text += f"• Text: IG-DiET agreement score: {overlap:.4f}\n"
            if overlap > 0.5:
                summary_text += "  → High agreement: both methods identify similar important tokens\n"
            else:
                summary_text += "  → Low agreement: DiET may capture different discriminative features\n"
        
        ax5.text(0.5, 0.5, summary_text, transform=ax5.transAxes,
                fontsize=12, verticalalignment='center', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        
        if save_name:
            save_path = os.path.join(self.output_dir, f"{save_name}.{self.config.save_format}")
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def generate_html_report(
        self,
        results: Dict[str, Any],
        report_title: str = "XAI Comparison Report",
        save_name: str = "comparison_report"
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
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2196F3;
            border-bottom: 3px solid #2196F3;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #4CAF50;
            margin-top: 30px;
        }}
        .section {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        .metric-table th, .metric-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .metric-table th {{
            background-color: #4CAF50;
            color: white;
        }}
        .metric-table tr:hover {{
            background-color: #f5f5f5;
        }}
        .highlight {{
            background-color: #e8f5e9;
            font-weight: bold;
        }}
        .summary-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-top: 30px;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            margin-right: 5px;
        }}
        .badge-gradcam {{ background-color: #2196F3; color: white; }}
        .badge-diet {{ background-color: #4CAF50; color: white; }}
        .badge-ig {{ background-color: #FF9800; color: white; }}
    </style>
</head>
<body>
    <h1>🔬 {report_title}</h1>
    <p>Generated: {np.datetime64('now')}</p>
"""
        
        # Image section
        if 'image_experiments' in results:
            img = results['image_experiments']
            html += """
    <div class="section">
        <h2>📸 Image Classification (CIFAR-10)</h2>
        <p>Comparison of <span class="badge badge-gradcam">GradCAM</span> vs 
           <span class="badge badge-diet">DiET</span></p>
        <table class="metric-table">
            <tr>
                <th>Metric</th>
                <th>GradCAM</th>
                <th>DiET</th>
                <th>Winner</th>
            </tr>
"""
            
            gradcam_score = img.get('gradcam_mean_score', 0)
            diet_score = img.get('diet_mean_score', 0)
            winner = 'DiET' if diet_score > gradcam_score else 'GradCAM'
            
            baseline_acc = img.get('baseline_accuracy', None)
            diet_acc = img.get('diet_accuracy', None)
            baseline_acc_str = f"{baseline_acc:.2f}%" if baseline_acc is not None else "N/A"
            diet_acc_str = f"{diet_acc:.2f}%" if diet_acc is not None else "N/A"
            
            html += f"""
            <tr>
                <td>Pixel Perturbation Score</td>
                <td>{gradcam_score:.4f}</td>
                <td>{diet_score:.4f}</td>
                <td class="highlight">{winner}</td>
            </tr>
            <tr>
                <td>Baseline Accuracy</td>
                <td colspan="2">{baseline_acc_str}</td>
                <td>-</td>
            </tr>
            <tr>
                <td>After DiET Accuracy</td>
                <td colspan="2">{diet_acc_str}</td>
                <td>-</td>
            </tr>
        </table>
    </div>
"""
        
        # Text section
        if 'text_experiments' in results:
            txt = results['text_experiments']
            
            ig_overlap = txt.get('ig_diet_overlap', None)
            ig_overlap_str = f"{ig_overlap:.4f}" if ig_overlap is not None else "N/A"
            txt_baseline_acc = txt.get('baseline_accuracy', None)
            txt_baseline_acc_str = f"{txt_baseline_acc:.2f}%" if txt_baseline_acc is not None else "N/A"
            
            html += f"""
    <div class="section">
        <h2>📝 Text Classification (SST-2)</h2>
        <p>Comparison of <span class="badge badge-ig">Integrated Gradients</span> vs 
           <span class="badge badge-diet">DiET</span></p>
        <table class="metric-table">
            <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Interpretation</th>
            </tr>
            <tr>
                <td>IG-DiET Token Overlap</td>
                <td>{ig_overlap_str}</td>
                <td>{"High agreement" if (ig_overlap or 0) > 0.5 else "Methods differ"}</td>
            </tr>
            <tr>
                <td>Baseline Accuracy</td>
                <td>{txt_baseline_acc_str}</td>
                <td>-</td>
            </tr>
            <tr>
                <td>Samples Compared</td>
                <td>{txt.get('samples_compared', 'N/A')}</td>
                <td>-</td>
            </tr>
        </table>
    </div>
"""
        
        # Summary
        html += """
    <div class="summary-box">
        <h2>📊 Key Findings</h2>
        <ul>
"""
        
        if 'image_experiments' in results:
            img = results['image_experiments']
            if img.get('diet_better', False):
                html += f"<li>✅ DiET improves image attribution quality by {img.get('improvement', 0):.4f}</li>"
            else:
                html += "<li>ℹ️ GradCAM performs adequately for this task</li>"
        
        if 'text_experiments' in results:
            txt = results['text_experiments']
            overlap = txt.get('ig_diet_overlap', 0)
            if overlap > 0.5:
                html += f"<li>✅ High agreement ({overlap:.2f}) between IG and DiET for text</li>"
            else:
                html += f"<li>🔍 DiET identifies different features than IG (overlap: {overlap:.2f})</li>"
        
        html += """
        </ul>
    </div>
</body>
</html>
"""
        
        save_path = os.path.join(self.output_dir, f"{save_name}.html")
        with open(save_path, 'w') as f:
            f.write(html)
        
        print(f"Saved HTML report: {save_path}")
        return save_path


# Convenience functions for quick plotting

def plot_metric_comparison(
    results: Dict[str, Dict[str, float]],
    output_dir: str = "./outputs",
    **kwargs
) -> plt.Figure:
    """Convenience function for quick metric comparison plot."""
    visualizer = ComparisonVisualizer(output_dir=output_dir)
    return visualizer.plot_metric_comparison_bar(results, **kwargs)


def plot_attribution_comparison(
    image: np.ndarray,
    attributions: Dict[str, np.ndarray],
    output_dir: str = "./outputs",
    **kwargs
) -> plt.Figure:
    """Convenience function for quick attribution comparison plot."""
    visualizer = ComparisonVisualizer(output_dir=output_dir)
    return visualizer.plot_image_attribution_comparison(image, attributions, **kwargs)


def create_comparison_report(
    results: Dict[str, Any],
    output_dir: str = "./outputs",
    **kwargs
) -> str:
    """Convenience function for generating HTML report."""
    visualizer = ComparisonVisualizer(output_dir=output_dir)
    return visualizer.generate_html_report(results, **kwargs)
