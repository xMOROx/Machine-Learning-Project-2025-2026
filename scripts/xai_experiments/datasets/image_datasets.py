"""Image dataset utilities for XAI comparison framework.

Supported datasets:
- CIFAR-10: 10 classes, 32x32 color images
- CIFAR-100: 100 classes, 32x32 color images
- SVHN: 10 classes (digits 0-9), 32x32 color images from street view
- Fashion-MNIST: 10 classes, grayscale images resized to 32x32
"""

from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Optional, Dict, Any


SUPPORTED_IMAGE_DATASETS = ["cifar10", "cifar100", "svhn", "fashion_mnist"]


class BaseImageDataset:
    """Base class for image datasets."""

    def __init__(
        self,
        data_dir: str = "./data",
        max_samples: Optional[int] = None,
        image_size: int = 32,
    ):
        self.data_dir = data_dir
        self.max_samples = max_samples
        self.image_size = image_size
        self.train_dataset = None
        self.test_dataset = None
        self.num_classes = 10
        self.class_names = []

    def get_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        """Get train and test transforms."""
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(self.image_size, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        return train_transform, test_transform

    def get_dataloaders(
        self, batch_size: int = 64, num_workers: int = 4
    ) -> Tuple[DataLoader, DataLoader]:
        """Get train and test dataloaders."""
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        return train_loader, test_loader

    def get_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        return {
            "name": self.__class__.__name__,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "train_size": len(self.train_dataset) if self.train_dataset else 0,
            "test_size": len(self.test_dataset) if self.test_dataset else 0,
            "image_size": self.image_size,
        }


class CIFAR10Dataset(BaseImageDataset):
    """CIFAR-10 dataset wrapper.

    10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    """

    def __init__(
        self,
        data_dir: str = "./data",
        max_samples: Optional[int] = None,
        image_size: int = 32,
    ):
        super().__init__(data_dir, max_samples, image_size)
        self.num_classes = 10
        self.class_names = [
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
        self._load_data()

    def _load_data(self):
        train_transform, test_transform = self.get_transforms()

        self.train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=train_transform,
        )

        self.test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=test_transform,
        )

        if self.max_samples:
            indices = list(range(min(self.max_samples, len(self.train_dataset))))
            self.train_dataset = Subset(self.train_dataset, indices)


class CIFAR100Dataset(BaseImageDataset):
    """CIFAR-100 dataset wrapper.

    100 classes organized into 20 superclasses.
    """

    def __init__(
        self,
        data_dir: str = "./data",
        max_samples: Optional[int] = None,
        image_size: int = 32,
    ):
        super().__init__(data_dir, max_samples, image_size)
        self.num_classes = 100
        # CIFAR-100 has 100 fine classes
        self.class_names = [f"class_{i}" for i in range(100)]
        self._load_data()

    def _load_data(self):
        train_transform, test_transform = self.get_transforms()

        self.train_dataset = torchvision.datasets.CIFAR100(
            root=self.data_dir,
            train=True,
            download=True,
            transform=train_transform,
        )

        self.test_dataset = torchvision.datasets.CIFAR100(
            root=self.data_dir,
            train=False,
            download=True,
            transform=test_transform,
        )

        if self.max_samples:
            indices = list(range(min(self.max_samples, len(self.train_dataset))))
            self.train_dataset = Subset(self.train_dataset, indices)


class SVHNDataset(BaseImageDataset):
    """Street View House Numbers (SVHN) dataset wrapper.

    10 classes: digits 0-9 from Google Street View.
    """

    def __init__(
        self,
        data_dir: str = "./data",
        max_samples: Optional[int] = None,
        image_size: int = 32,
    ):
        super().__init__(data_dir, max_samples, image_size)
        self.num_classes = 10
        self.class_names = [str(i) for i in range(10)]
        self._load_data()

    def _load_data(self):
        train_transform, test_transform = self.get_transforms()

        self.train_dataset = torchvision.datasets.SVHN(
            root=self.data_dir,
            split="train",
            download=True,
            transform=train_transform,
        )

        self.test_dataset = torchvision.datasets.SVHN(
            root=self.data_dir,
            split="test",
            download=True,
            transform=test_transform,
        )

        if self.max_samples:
            indices = list(range(min(self.max_samples, len(self.train_dataset))))
            self.train_dataset = Subset(self.train_dataset, indices)


class FashionMNISTDataset(BaseImageDataset):
    """Fashion-MNIST dataset wrapper.

    10 classes: T-shirt/top, Trouser, Pullover, Dress, Coat,
                Sandal, Shirt, Sneaker, Bag, Ankle boot
    Grayscale images converted to 3-channel and resized to 32x32.
    """

    def __init__(
        self,
        data_dir: str = "./data",
        max_samples: Optional[int] = None,
        image_size: int = 32,
    ):
        super().__init__(data_dir, max_samples, image_size)
        self.num_classes = 10
        self.class_names = [
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ]
        self._load_data()

    def get_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        """Get transforms for grayscale to RGB conversion."""
        train_transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        return train_transform, test_transform

    def _load_data(self):
        train_transform, test_transform = self.get_transforms()

        self.train_dataset = torchvision.datasets.FashionMNIST(
            root=self.data_dir,
            train=True,
            download=True,
            transform=train_transform,
        )

        self.test_dataset = torchvision.datasets.FashionMNIST(
            root=self.data_dir,
            train=False,
            download=True,
            transform=test_transform,
        )

        if self.max_samples:
            indices = list(range(min(self.max_samples, len(self.train_dataset))))
            self.train_dataset = Subset(self.train_dataset, indices)


def get_image_dataset(
    name: str,
    data_dir: str = "./data",
    max_samples: Optional[int] = None,
) -> BaseImageDataset:
    """Factory function to get image dataset by name.

    Args:
        name: Dataset name (cifar10, cifar100, svhn, fashion_mnist)
        data_dir: Directory to store/load data
        max_samples: Maximum number of training samples

    Returns:
        Dataset wrapper instance

    Raises:
        ValueError: If dataset name is not supported
    """
    name = name.lower()

    if name == "cifar10":
        return CIFAR10Dataset(data_dir, max_samples)
    elif name == "cifar100":
        return CIFAR100Dataset(data_dir, max_samples)
    elif name == "svhn":
        return SVHNDataset(data_dir, max_samples)
    elif name in ["fashion_mnist", "fashionmnist", "fmnist"]:
        return FashionMNISTDataset(data_dir, max_samples)
    else:
        raise ValueError(
            f"Unsupported image dataset: {name}. Supported: {SUPPORTED_IMAGE_DATASETS}"
        )
