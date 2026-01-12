"""Dataset utilities for XAI comparison framework.

This module provides support for multiple datasets:

Image Datasets:
- CIFAR-10: 10-class image classification (32x32)
- CIFAR-100: 100-class image classification (32x32)
- SVHN: Street View House Numbers (32x32)
- Fashion-MNIST: Fashion product images (28x28, resized to 32x32)

Text Datasets:
- SST-2: Stanford Sentiment Treebank (binary sentiment)
- IMDB: Movie reviews (binary sentiment)
- AG News: News classification (4 classes)
"""

from .image_datasets import (
    get_image_dataset,
    CIFAR10Dataset,
    CIFAR100Dataset,
    SVHNDataset,
    FashionMNISTDataset,
    SUPPORTED_IMAGE_DATASETS,
)

from .text_datasets import (
    get_text_dataset,
    SST2Dataset,
    IMDBDataset,
    AGNewsDataset,
    SUPPORTED_TEXT_DATASETS,
)

__all__ = [
    # Image datasets
    "get_image_dataset",
    "CIFAR10Dataset",
    "CIFAR100Dataset",
    "SVHNDataset",
    "FashionMNISTDataset",
    "SUPPORTED_IMAGE_DATASETS",
    # Text datasets
    "get_text_dataset",
    "SST2Dataset",
    "IMDBDataset",
    "AGNewsDataset",
    "SUPPORTED_TEXT_DATASETS",
]
