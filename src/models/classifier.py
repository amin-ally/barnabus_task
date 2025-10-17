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
        model_name: str = "distilbert-base-multilingual-cased",
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
            nn.LayerNorm(hidden_size),
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
        """
        Forward pass.
        Uses the [CLS] token's output for classification.
        Uses mean pooling of all token outputs for generating similarity embeddings.
        """

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # --- CLS TOKEN FOR CLASSIFICATION ---
        # The output corresponding to the [CLS] token is the first one in the sequence.
        # Shape: (batch_size, hidden_size)
        cls_output = last_hidden_state[:, 0]
        logits = self.classifier(cls_output)
        probs = torch.softmax(logits, dim=-1)

        # Calculate confidence (max probability)
        confidence, predicted = torch.max(probs, dim=-1)

        # --- MEAN POOLING FOR SIMILARITY EMBEDDINGS ---
        embeddings = None
        if return_embeddings:
            # Step 1: Get the attention mask in the right shape for broadcasting.
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            )
            # Step 2: Zero out the embeddings of padding tokens.
            sum_embeddings = last_hidden_state * input_mask_expanded
            # Step 3: Sum the vectors along the sequence dimension.
            sum_embeddings = torch.sum(sum_embeddings, 1)
            # Step 4: Count the number of actual tokens in each sentence.
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            # Step 5: Calculate the average to get the mean-pooled sentence representation.
            mean_pooled_output = sum_embeddings / sum_mask

            # Normalize the embeddings for use in cosine similarity searches
            embeddings = torch.nn.functional.normalize(mean_pooled_output, p=2, dim=1)

        return {
            "logits": logits,
            "probabilities": probs,
            "predictions": predicted,
            "confidence": confidence,
            "embeddings": embeddings,
            "hidden_states": outputs.last_hidden_state,
            "attentions": outputs.attentions,
        }
