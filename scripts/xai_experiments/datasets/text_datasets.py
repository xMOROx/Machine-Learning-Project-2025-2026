"""Text dataset utilities for XAI comparison framework.

Supported datasets:
- SST-2: Stanford Sentiment Treebank (binary sentiment, from GLUE)
- IMDB: Movie reviews (binary sentiment)
- AG News: News classification (4 classes)
"""

import torch
from typing import Tuple, Optional, Dict, Any, List


SUPPORTED_TEXT_DATASETS = ["sst2", "imdb", "ag_news"]


class BaseTextDataset:
    """Base class for text datasets."""

    def __init__(
        self,
        data_dir: str = "./data",
        max_samples: Optional[int] = None,
        max_length: int = 128,
        model_name: str = "bert-base-uncased",
    ):
        self.data_dir = data_dir
        self.max_samples = max_samples
        self.max_length = max_length
        self.model_name = model_name
        self.num_classes = 2
        self.class_names = []
        self.tokenizer = None

        self.train_texts = []
        self.train_labels = []
        self.val_texts = []
        self.val_labels = []

    def _init_tokenizer(self):
        """Initialize tokenizer."""
        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        except ImportError:
            raise ImportError(
                "transformers library required. Install with: pip install transformers"
            )

    def tokenize(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize texts.

        Args:
            texts: List of text strings

        Returns:
            Tuple of (input_ids, attention_mask)
        """
        if self.tokenizer is None:
            self._init_tokenizer()

        encoded = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return encoded["input_ids"], encoded["attention_mask"]

    def get_train_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get tokenized training data.

        Returns:
            Tuple of (input_ids, attention_mask, labels)
        """
        input_ids, attention_mask = self.tokenize(self.train_texts)
        labels = torch.tensor(self.train_labels)
        return input_ids, attention_mask, labels

    def get_val_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get tokenized validation data.

        Returns:
            Tuple of (input_ids, attention_mask, labels)
        """
        input_ids, attention_mask = self.tokenize(self.val_texts)
        labels = torch.tensor(self.val_labels)
        return input_ids, attention_mask, labels

    def get_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        return {
            "name": self.__class__.__name__,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "train_size": len(self.train_texts),
            "val_size": len(self.val_texts),
            "max_length": self.max_length,
        }


class SST2Dataset(BaseTextDataset):
    """Stanford Sentiment Treebank v2 (SST-2) dataset.

    Binary sentiment classification from movie reviews.
    Part of the GLUE benchmark.
    """

    def __init__(
        self,
        data_dir: str = "./data",
        max_samples: Optional[int] = None,
        max_length: int = 128,
        model_name: str = "bert-base-uncased",
    ):
        super().__init__(data_dir, max_samples, max_length, model_name)
        self.num_classes = 2
        self.class_names = ["negative", "positive"]
        self._load_data()

    def _load_data(self):
        """Load SST-2 dataset from Hugging Face."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets library required. Install with: pip install datasets"
            )

        dataset = load_dataset("glue", "sst2")

        train_data = dataset["train"]
        val_data = dataset["validation"]

        # Limit samples if specified
        if self.max_samples:
            train_data = train_data.select(
                range(min(self.max_samples, len(train_data)))
            )

        self.train_texts = train_data["sentence"]
        self.train_labels = train_data["label"]
        self.val_texts = val_data["sentence"]
        self.val_labels = val_data["label"]


class IMDBDataset(BaseTextDataset):
    """IMDB movie reviews dataset.

    Binary sentiment classification from movie reviews.
    50,000 reviews (25k train, 25k test).
    """

    def __init__(
        self,
        data_dir: str = "./data",
        max_samples: Optional[int] = None,
        max_length: int = 256,  # IMDB reviews are longer
        model_name: str = "bert-base-uncased",
    ):
        super().__init__(data_dir, max_samples, max_length, model_name)
        self.num_classes = 2
        self.class_names = ["negative", "positive"]
        self._load_data()

    def _load_data(self):
        """Load IMDB dataset from Hugging Face."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets library required. Install with: pip install datasets"
            )

        dataset = load_dataset("imdb")

        train_data = dataset["train"]
        test_data = dataset["test"]

        # Limit samples if specified
        if self.max_samples:
            train_data = train_data.select(
                range(min(self.max_samples, len(train_data)))
            )
            test_data = test_data.select(
                range(min(self.max_samples // 2, len(test_data)))
            )

        self.train_texts = train_data["text"]
        self.train_labels = train_data["label"]
        self.val_texts = test_data["text"]
        self.val_labels = test_data["label"]


class AGNewsDataset(BaseTextDataset):
    """AG News dataset.

    4-class news topic classification:
    - World (0)
    - Sports (1)
    - Business (2)
    - Sci/Tech (3)
    """

    def __init__(
        self,
        data_dir: str = "./data",
        max_samples: Optional[int] = None,
        max_length: int = 128,
        model_name: str = "bert-base-uncased",
    ):
        super().__init__(data_dir, max_samples, max_length, model_name)
        self.num_classes = 4
        self.class_names = ["World", "Sports", "Business", "Sci/Tech"]
        self._load_data()

    def _load_data(self):
        """Load AG News dataset from Hugging Face."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets library required. Install with: pip install datasets"
            )

        dataset = load_dataset("ag_news")

        train_data = dataset["train"]
        test_data = dataset["test"]

        # Limit samples if specified
        if self.max_samples:
            train_data = train_data.select(
                range(min(self.max_samples, len(train_data)))
            )
            test_data = test_data.select(
                range(min(self.max_samples // 4, len(test_data)))
            )

        self.train_texts = train_data["text"]
        self.train_labels = train_data["label"]
        self.val_texts = test_data["text"]
        self.val_labels = test_data["label"]


def get_text_dataset(
    name: str,
    data_dir: str = "./data",
    max_samples: Optional[int] = None,
    max_length: int = 128,
    model_name: str = "bert-base-uncased",
) -> BaseTextDataset:
    """Factory function to get text dataset by name.

    Args:
        name: Dataset name (sst2, imdb, ag_news)
        data_dir: Directory to store/load data
        max_samples: Maximum number of training samples
        max_length: Maximum sequence length for tokenization
        model_name: Pretrained model name for tokenizer

    Returns:
        Dataset wrapper instance

    Raises:
        ValueError: If dataset name is not supported
    """
    name = name.lower()

    if name == "sst2":
        return SST2Dataset(data_dir, max_samples, max_length, model_name)
    elif name == "imdb":
        # IMDB reviews are longer, use 256 by default
        return IMDBDataset(data_dir, max_samples, max(max_length, 256), model_name)
    elif name in ["ag_news", "agnews"]:
        return AGNewsDataset(data_dir, max_samples, max_length, model_name)
    else:
        raise ValueError(
            f"Unsupported text dataset: {name}. Supported: {SUPPORTED_TEXT_DATASETS}"
        )
