# src/models/classifier.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Tuple, Optional
import numpy as np


class HateSpeechClassifier(nn.Module):
    """Multi-class classifier for hate speech detection"""

    def __init__(
        self,
        model_name: str = "distilbert-base-multilingual-cased",  # Updated to match config.yaml
        num_classes: int = 3,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes

        # Load pretrained model
        self.bert = AutoModel.from_pretrained(model_name, output_attentions=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Get hidden size
        hidden_size = self.bert.config.hidden_size

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_classes),
        )

        # For embeddings
        self.embedding_layer = nn.Linear(
            hidden_size, 384
        )  # Reduce dimension for storage

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass"""

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use CLS token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]

        # Get embeddings if needed
        embeddings = None
        if return_embeddings:
            embeddings = self.embedding_layer(pooled_output)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # Classification
        logits = self.classifier(pooled_output)
        probs = torch.softmax(logits, dim=-1)

        # Calculate confidence (max probability)
        confidence, predicted = torch.max(probs, dim=-1)

        return {
            "logits": logits,
            "probabilities": probs,
            "predictions": predicted,
            "confidence": confidence,
            "embeddings": embeddings,
            "hidden_states": outputs.last_hidden_state,
            "attentions": outputs.attentions,
        }
