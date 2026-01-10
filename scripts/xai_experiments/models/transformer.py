"""Transformer Models for Language Classification Tasks."""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class BertForSequenceClassificationWithIG(nn.Module):
    """BERT model wrapper for sequence classification with Integrated Gradients support.

    This wrapper provides a clean interface for Integrated Gradients attribution
    by exposing the embedding layer and providing methods to compute attributions.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 2,
        device: str = "cuda",
    ):
        super().__init__()

        from transformers import BertForSequenceClassification, BertTokenizer

        self.device = device
        self.model_name = model_name
        self.num_labels = num_labels

        self.model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        self.model = self.model.to(device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass returning logits only."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return outputs.logits

    def forward_with_embeddings(
        self, input_embeds: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass using embedding inputs (for Integrated Gradients)."""
        outputs = self.model(inputs_embeds=input_embeds, attention_mask=attention_mask)
        return outputs.logits

    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get embeddings from input_ids."""
        return self.model.bert.embeddings.word_embeddings(input_ids)

    def encode_text(
        self, text: str, max_length: int = 128
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text to input_ids and attention_mask."""
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return (
            encoding["input_ids"].to(self.device),
            encoding["attention_mask"].to(self.device),
        )

    def decode_tokens(self, input_ids: torch.Tensor) -> list:
        """Decode input_ids to tokens."""
        return self.tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())

    def train_mode(self):
        """Set model to training mode."""
        self.model.train()

    def eval_mode(self):
        """Set model to evaluation mode."""
        self.model.eval()

    def get_parameters(self):
        """Get model parameters for optimizer."""
        return self.model.parameters()

    def save_model(self, path: str):
        """Save model and tokenizer."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_model(self, path: str):
        """Load model and tokenizer from path."""
        from transformers import BertForSequenceClassification, BertTokenizer

        self.model = BertForSequenceClassification.from_pretrained(path).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(path)

    def get_state_dict(self):
        """Get model state dict for checkpointing."""
        return self.model.state_dict()

    def load_state_dict_from_checkpoint(self, state_dict):
        """Load model state dict from checkpoint."""
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
