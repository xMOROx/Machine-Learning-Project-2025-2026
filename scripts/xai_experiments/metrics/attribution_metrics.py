"""Comprehensive Attribution Metrics for XAI Evaluation.

This module implements standard evaluation metrics for comparing XAI methods:
1. Pixel Perturbation - Remove/mask important regions and measure accuracy drop
2. Insertion/Deletion - Progressively insert/delete important pixels
3. AOPC (Area Over Perturbation Curve) - Average accuracy drop over perturbations
4. Faithfulness Correlation - Correlation between attribution and model sensitivity
5. Top-k Token Overlap - For text attribution comparison

Reference papers:
- Samek et al., "Evaluating the Visualization of What a Deep Neural Network has Learned"
- Petsiuk et al., "RISE: Randomized Input Sampling for Explanation of Black-box Models"
- Bach et al., "On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation"
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class MetricResult:
    """Container for metric computation results."""

    name: str
    value: float
    std: Optional[float] = None
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {"name": self.name, "value": self.value}
        if self.std is not None:
            result["std"] = self.std
        if self.details is not None:
            result["details"] = self.details
        return result


class BaseMetric(ABC):
    """Abstract base class for attribution metrics."""

    @abstractmethod
    def compute(self, *args, **kwargs) -> MetricResult:
        """Compute the metric."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return metric name."""
        pass


class PixelPerturbation(BaseMetric):
    """Pixel Perturbation Metric for Image Attributions.

    Evaluates attributions by masking top-k% important pixels and
    measuring the accuracy drop. Higher initial accuracy with important
    pixels kept indicates better attribution quality.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda",
        percentages: List[int] = [5, 10, 20, 30, 50, 70, 90],
        perturbation_type: str = "keep",  # "keep" or "remove"
    ):
        """Initialize Pixel Perturbation metric.

        Args:
            model: The model to evaluate
            device: Device for computation
            percentages: List of percentages to evaluate
            perturbation_type: "keep" to keep top pixels, "remove" to remove them
        """
        self.model = model
        self.device = device
        self.percentages = percentages
        self.perturbation_type = perturbation_type
        self.model.eval()

    @property
    def name(self) -> str:
        return f"PixelPerturbation_{self.perturbation_type}"

    def compute(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        attribution_maps: List[np.ndarray],
        baseline_value: float = 0.0,
    ) -> MetricResult:
        """Compute pixel perturbation scores.

        Args:
            images: Image tensor (N, C, H, W)
            labels: True labels (N,)
            attribution_maps: List of attribution maps
            baseline_value: Value to use for masked pixels

        Returns:
            MetricResult with accuracy at each perturbation level
        """
        results_per_pct = {p: [] for p in self.percentages}

        with torch.no_grad():
            for _, (image, label, attr_map) in enumerate(
                zip(images, labels, attribution_maps)
            ):
                image = image.unsqueeze(0).to(self.device)

                # Resize attribution to image size
                if isinstance(attr_map, np.ndarray):
                    attr_tensor = torch.tensor(attr_map).float()
                else:
                    attr_tensor = attr_map.float()

                if attr_tensor.dim() == 2:
                    attr_tensor = attr_tensor.unsqueeze(0).unsqueeze(0)
                elif attr_tensor.dim() == 3:
                    attr_tensor = attr_tensor.unsqueeze(0)

                attr_resized = (
                    F.interpolate(
                        attr_tensor,
                        size=(image.shape[2], image.shape[3]),
                        mode="bilinear",
                        align_corners=False,
                    )
                    .squeeze()
                    .numpy()
                )

                for pct in self.percentages:
                    if self.perturbation_type == "keep":
                        threshold = np.percentile(attr_resized, 100 - pct)
                        mask = (attr_resized >= threshold).astype(np.float32)
                    else:  # remove
                        threshold = np.percentile(attr_resized, 100 - pct)
                        mask = (attr_resized < threshold).astype(np.float32)

                    mask = torch.tensor(mask).unsqueeze(0).unsqueeze(0).to(self.device)

                    # Apply mask
                    masked_image = image * mask + baseline_value * (1 - mask)

                    # Get prediction
                    output = self.model(masked_image)
                    pred = output.argmax(1).item()

                    correct = 1 if pred == label.item() else 0
                    results_per_pct[pct].append(correct)

        # Compute mean accuracy per percentage
        accuracies = {p: np.mean(results_per_pct[p]) for p in self.percentages}
        mean_score = np.mean(list(accuracies.values()))

        return MetricResult(
            name=self.name,
            value=mean_score,
            std=np.std(list(accuracies.values())),
            details={"accuracies_by_percentage": accuracies},
        )


class InsertionDeletion(BaseMetric):
    """Insertion/Deletion Metric for Image Attributions.

    Deletion: Start with original image, progressively mask most important pixels.
              Faster accuracy drop = better attribution.
    Insertion: Start with baseline, progressively reveal most important pixels.
               Faster accuracy increase = better attribution.

    Reference: Petsiuk et al., "RISE: Randomized Input Sampling for
               Explanation of Black-box Models"
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda",
        num_steps: int = 100,
        substrate_fn: str = "blur",  # "blur", "mean", or "zero"
    ):
        """Initialize Insertion/Deletion metric.

        Args:
            model: The model to evaluate
            device: Device for computation
            num_steps: Number of insertion/deletion steps
            substrate_fn: Background generation method
        """
        self.model = model
        self.device = device
        self.num_steps = num_steps
        self.substrate_fn = substrate_fn
        self.model.eval()

    @property
    def name(self) -> str:
        return "InsertionDeletion"

    def _get_substrate(self, image: torch.Tensor) -> torch.Tensor:
        """Generate substrate (background) for masked regions."""
        if self.substrate_fn == "blur":
            # Use Gaussian blur
            from torchvision.transforms import GaussianBlur

            blur = GaussianBlur(kernel_size=51, sigma=50.0)
            return blur(image)
        elif self.substrate_fn == "mean":
            return torch.full_like(image, image.mean())
        else:  # zero
            return torch.zeros_like(image)

    def compute(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        attribution_maps: List[np.ndarray],
    ) -> MetricResult:
        """Compute insertion and deletion curves.

        Args:
            images: Image tensor (N, C, H, W)
            labels: True labels (N,)
            attribution_maps: List of attribution maps

        Returns:
            MetricResult with AUC scores for insertion and deletion
        """
        insertion_scores = []
        deletion_scores = []

        with torch.no_grad():
            for image, label, attr_map in zip(images, labels, attribution_maps):
                image = image.unsqueeze(0).to(self.device)
                substrate = self._get_substrate(image)

                # Resize and flatten attribution
                if isinstance(attr_map, np.ndarray):
                    attr_tensor = torch.tensor(attr_map).float()
                else:
                    attr_tensor = attr_map.float()

                if attr_tensor.dim() == 2:
                    attr_tensor = attr_tensor.unsqueeze(0).unsqueeze(0)

                attr_resized = F.interpolate(
                    attr_tensor,
                    size=(image.shape[2], image.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()

                # Get sorted indices (highest to lowest importance)
                h, w = attr_resized.shape
                num_pixels = h * w
                flat_attr = attr_resized.flatten()
                sorted_indices = torch.argsort(flat_attr, descending=True)

                # Calculate step size
                pixels_per_step = max(1, num_pixels // self.num_steps)

                insertion_probs = []
                deletion_probs = []

                for step in range(self.num_steps + 1):
                    num_revealed = min(step * pixels_per_step, num_pixels)

                    # Create mask
                    mask = torch.zeros(num_pixels, device=self.device)
                    if num_revealed > 0:
                        mask[sorted_indices[:num_revealed]] = 1.0
                    mask = mask.view(1, 1, h, w)

                    # Insertion: start from substrate, add pixels
                    insertion_img = substrate * (1 - mask) + image * mask
                    ins_output = self.model(insertion_img)
                    ins_prob = F.softmax(ins_output, dim=1)[0, label].item()
                    insertion_probs.append(ins_prob)

                    # Deletion: start from original, remove pixels
                    deletion_img = image * (1 - mask) + substrate * mask
                    del_output = self.model(deletion_img)
                    del_prob = F.softmax(del_output, dim=1)[0, label].item()
                    deletion_probs.append(del_prob)

                # Compute AUC using trapezoidal rule
                insertion_auc = np.trapz(insertion_probs) / len(insertion_probs)
                deletion_auc = np.trapz(deletion_probs) / len(deletion_probs)

                insertion_scores.append(insertion_auc)
                deletion_scores.append(deletion_auc)

        mean_insertion = np.mean(insertion_scores)
        mean_deletion = np.mean(deletion_scores)

        return MetricResult(
            name=self.name,
            value=mean_insertion - mean_deletion,  # Higher is better
            details={
                "insertion_auc": mean_insertion,
                "insertion_std": np.std(insertion_scores),
                "deletion_auc": mean_deletion,
                "deletion_std": np.std(deletion_scores),
                "insertion_scores": insertion_scores,
                "deletion_scores": deletion_scores,
            },
        )


class AOPC(BaseMetric):
    """Area Over Perturbation Curve (AOPC) Metric.

    Measures the average change in prediction probability when
    progressively removing the most important features.

    Reference: Samek et al., "Evaluating the Visualization of What a
               Deep Neural Network has Learned"
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda",
        num_steps: int = 10,
        patch_size: int = 4,
    ):
        """Initialize AOPC metric.

        Args:
            model: The model to evaluate
            device: Device for computation
            num_steps: Number of perturbation steps
            patch_size: Size of patches to perturb
        """
        self.model = model
        self.device = device
        self.num_steps = num_steps
        self.patch_size = patch_size
        self.model.eval()

    @property
    def name(self) -> str:
        return "AOPC"

    def compute(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        attribution_maps: List[np.ndarray],
    ) -> MetricResult:
        """Compute AOPC score.

        Args:
            images: Image tensor (N, C, H, W)
            labels: True labels (N,)
            attribution_maps: List of attribution maps

        Returns:
            MetricResult with AOPC score
        """
        aopc_scores = []

        with torch.no_grad():
            for image, label, attr_map in zip(images, labels, attribution_maps):
                image = image.unsqueeze(0).to(self.device)

                # Get initial probability
                output = self.model(image)
                probs = F.softmax(output, dim=1)
                initial_prob = probs[0, label].item()

                # Resize attribution
                if isinstance(attr_map, np.ndarray):
                    attr_tensor = torch.tensor(attr_map).float()
                else:
                    attr_tensor = attr_map.float()

                if attr_tensor.dim() == 2:
                    attr_tensor = attr_tensor.unsqueeze(0).unsqueeze(0)

                attr_resized = (
                    F.interpolate(
                        attr_tensor,
                        size=(image.shape[2], image.shape[3]),
                        mode="bilinear",
                        align_corners=False,
                    )
                    .squeeze()
                    .numpy()
                )

                # Compute patch importance
                h, w = attr_resized.shape
                num_patches_h = h // self.patch_size
                num_patches_w = w // self.patch_size

                patch_importance = []
                for ph in range(num_patches_h):
                    for pw in range(num_patches_w):
                        patch = attr_resized[
                            ph * self.patch_size : (ph + 1) * self.patch_size,
                            pw * self.patch_size : (pw + 1) * self.patch_size,
                        ]
                        patch_importance.append((ph, pw, patch.mean()))

                # Sort patches by importance (descending)
                patch_importance.sort(key=lambda x: x[2], reverse=True)

                # Progressively remove patches
                perturbed_image = image.clone()
                prob_changes = []

                for step in range(min(self.num_steps, len(patch_importance))):
                    ph, pw, _ = patch_importance[step]

                    # Mask the patch (set to zero or mean)
                    perturbed_image[
                        :,
                        :,
                        ph * self.patch_size : (ph + 1) * self.patch_size,
                        pw * self.patch_size : (pw + 1) * self.patch_size,
                    ] = 0

                    # Get new probability
                    output = self.model(perturbed_image)
                    probs = F.softmax(output, dim=1)
                    new_prob = probs[0, label].item()

                    prob_changes.append(initial_prob - new_prob)

                # AOPC is the average probability drop
                aopc = np.mean(prob_changes) if prob_changes else 0
                aopc_scores.append(aopc)

        return MetricResult(
            name=self.name,
            value=np.mean(aopc_scores),
            std=np.std(aopc_scores),
            details={"per_sample_aopc": aopc_scores},
        )


class FaithfulnessCorrelation(BaseMetric):
    """Faithfulness Correlation Metric.

    Measures the correlation between attribution values and the change
    in model output when perturbing those features.

    Higher correlation = more faithful attribution.
    """

    def __init__(
        self, model: torch.nn.Module, device: str = "cuda", num_samples: int = 50
    ):
        """Initialize Faithfulness Correlation metric.

        Args:
            model: The model to evaluate
            device: Device for computation
            num_samples: Number of random perturbations per image
        """
        self.model = model
        self.device = device
        self.num_samples = num_samples
        self.model.eval()

    @property
    def name(self) -> str:
        return "FaithfulnessCorrelation"

    def compute(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        attribution_maps: List[np.ndarray],
    ) -> MetricResult:
        """Compute faithfulness correlation.

        Args:
            images: Image tensor (N, C, H, W)
            labels: True labels (N,)
            attribution_maps: List of attribution maps

        Returns:
            MetricResult with correlation score
        """
        correlations = []

        with torch.no_grad():
            for image, label, attr_map in zip(images, labels, attribution_maps):
                image = image.unsqueeze(0).to(self.device)

                # Get initial prediction
                output = self.model(image)
                probs = F.softmax(output, dim=1)
                initial_prob = probs[0, label].item()

                # Resize attribution
                if isinstance(attr_map, np.ndarray):
                    attr_tensor = torch.tensor(attr_map).float()
                else:
                    attr_tensor = attr_map.float()

                if attr_tensor.dim() == 2:
                    attr_tensor = attr_tensor.unsqueeze(0).unsqueeze(0)

                attr_resized = F.interpolate(
                    attr_tensor,
                    size=(image.shape[2], image.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()

                # Sample random perturbations
                h, w = attr_resized.shape
                attribution_sums = []
                prob_changes = []

                for _ in range(self.num_samples):
                    # Random mask (randomly remove 10-50% of pixels)
                    mask_ratio = np.random.uniform(0.1, 0.5)
                    mask = torch.rand(h, w, device=self.device) > mask_ratio
                    mask = mask.float().unsqueeze(0).unsqueeze(0)

                    # Apply mask
                    perturbed = image * mask

                    # Get prediction change
                    output = self.model(perturbed)
                    probs = F.softmax(output, dim=1)
                    new_prob = probs[0, label].item()
                    prob_change = initial_prob - new_prob

                    # Get sum of removed attributions
                    removed_mask = 1 - mask.squeeze()
                    attr_sum = (attr_resized * removed_mask).sum().item()

                    attribution_sums.append(attr_sum)
                    prob_changes.append(prob_change)

                # Compute Pearson correlation
                if len(attribution_sums) > 1:
                    corr = np.corrcoef(attribution_sums, prob_changes)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)

        mean_corr = np.mean(correlations) if correlations else 0

        return MetricResult(
            name=self.name,
            value=mean_corr,
            std=np.std(correlations) if correlations else 0,
            details={"per_sample_correlation": correlations},
        )


class TopKTokenOverlap(BaseMetric):
    """Top-k Token Overlap Metric for Text Attributions.

    Measures the overlap between the top-k most important tokens
    identified by two different attribution methods.
    """

    def __init__(self, k: int = 5):
        """Initialize Top-k Token Overlap metric.

        Args:
            k: Number of top tokens to compare
        """
        self.k = k

    @property
    def name(self) -> str:
        return f"TopK_TokenOverlap_k{self.k}"

    def compute(
        self,
        attributions_method1: List[np.ndarray],
        attributions_method2: List[np.ndarray],
        attention_masks: Optional[List[np.ndarray]] = None,
    ) -> MetricResult:
        """Compute top-k token overlap.

        Args:
            attributions_method1: Attributions from first method
            attributions_method2: Attributions from second method
            attention_masks: Optional attention masks to filter valid tokens

        Returns:
            MetricResult with overlap scores
        """
        overlaps = []

        for i, (attr1, attr2) in enumerate(
            zip(attributions_method1, attributions_method2)
        ):
            # Get valid token indices
            if attention_masks is not None:
                valid_mask = attention_masks[i] > 0
                attr1 = np.abs(attr1[valid_mask])
                attr2 = np.abs(attr2[valid_mask])
            else:
                attr1 = np.abs(attr1)
                attr2 = np.abs(attr2)

            # Get top-k indices
            k = min(self.k, len(attr1))
            top_k_1 = set(np.argsort(attr1)[-k:])
            top_k_2 = set(np.argsort(attr2)[-k:])

            # Compute overlap
            overlap = len(top_k_1 & top_k_2) / k
            overlaps.append(overlap)

        return MetricResult(
            name=self.name,
            value=np.mean(overlaps),
            std=np.std(overlaps),
            details={"per_sample_overlap": overlaps},
        )


class AttributionMetrics:
    """Unified interface for computing all attribution metrics.

    This class provides a convenient way to compute multiple metrics
    at once for comparing XAI methods.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda",
        modality: str = "image",  # "image" or "text"
    ):
        """Initialize AttributionMetrics.

        Args:
            model: The model to evaluate
            device: Device for computation
            modality: "image" or "text"
        """
        self.model = model
        self.device = device
        self.modality = modality

        if modality == "image":
            self.metrics = {
                "pixel_perturbation_keep": PixelPerturbation(
                    model, device, perturbation_type="keep"
                ),
                "pixel_perturbation_remove": PixelPerturbation(
                    model, device, perturbation_type="remove"
                ),
                "insertion_deletion": InsertionDeletion(model, device),
                "aopc": AOPC(model, device),
                "faithfulness": FaithfulnessCorrelation(model, device),
            }
        else:
            self.metrics = {
                "top_k_overlap_3": TopKTokenOverlap(k=3),
                "top_k_overlap_5": TopKTokenOverlap(k=5),
                "top_k_overlap_10": TopKTokenOverlap(k=10),
            }

    def compute_all(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        attribution_maps: List[np.ndarray],
        **kwargs,
    ) -> Dict[str, MetricResult]:
        """Compute all applicable metrics.

        Args:
            images: Input tensors
            labels: True labels
            attribution_maps: Attribution maps from XAI method
            **kwargs: Additional arguments for specific metrics

        Returns:
            Dictionary of metric names to MetricResult objects
        """
        results = {}

        for name, metric in self.metrics.items():
            try:
                if self.modality == "image":
                    result = metric.compute(images, labels, attribution_maps)
                else:
                    # For text metrics, need comparison attributions
                    if "comparison_attributions" in kwargs:
                        result = metric.compute(
                            attribution_maps,
                            kwargs["comparison_attributions"],
                            kwargs.get("attention_masks"),
                        )
                    else:
                        continue
                results[name] = result
            except Exception as e:
                print(f"Error computing {name}: {e}")
                continue

        return results

    def compute_comparison_matrix(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        method_attributions: Dict[str, List[np.ndarray]],
    ) -> Dict[str, Dict[str, MetricResult]]:
        """Compute metrics for multiple attribution methods.

        Args:
            images: Input tensors
            labels: True labels
            method_attributions: Dictionary mapping method names to attributions

        Returns:
            Nested dictionary: method_name -> metric_name -> MetricResult
        """
        results = {}

        for method_name, attributions in method_attributions.items():
            print(f"Computing metrics for {method_name}...")
            results[method_name] = self.compute_all(images, labels, attributions)

        return results

    def results_to_dataframe(self, results: Dict[str, Dict[str, MetricResult]]):
        """Convert results to a pandas DataFrame for easy visualization.

        Args:
            results: Results from compute_comparison_matrix

        Returns:
            pandas DataFrame with methods as rows and metrics as columns
        """
        import pandas as pd

        data = {}
        for method_name, metrics in results.items():
            data[method_name] = {}
            for metric_name, result in metrics.items():
                data[method_name][metric_name] = result.value
                if result.std is not None:
                    data[method_name][f"{metric_name}_std"] = result.std

        return pd.DataFrame(data).T
