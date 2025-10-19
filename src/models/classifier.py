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
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.onnx_export_mode = False  # Add this flag for ONNX export

        # Load pretrained model
        self.bert = AutoModel.from_pretrained(model_name, output_attentions=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if freeze_backbone:
            for param in self.bert.parameters():
                param.requires_grad = False

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

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        # This flag is now only used for the PyTorch dictionary output
        return_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        - Always computes logits and embeddings for ONNX tracing.
        - Returns a tuple for ONNX export mode.
        - Returns a dictionary for standard PyTorch mode.
        """

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # --- CLS TOKEN FOR CLASSIFICATION ---
        cls_output = last_hidden_state[:, 0]
        logits = self.classifier(cls_output)

        # --- MEAN POOLING FOR SIMILARITY EMBEDDINGS ---
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_pooled_output = sum_embeddings / sum_mask
        embeddings = torch.nn.functional.normalize(mean_pooled_output, p=2, dim=1)

        # --- ONNX Export Path ---
        if self.onnx_export_mode:
            return logits, outputs.attentions[-1], embeddings

        # --- Standard PyTorch Path ---
        probs = torch.softmax(logits, dim=-1)
        confidence, predicted = torch.max(probs, dim=-1)

        output_dict = {
            "logits": logits,
            "probabilities": probs,
            "predictions": predicted,
            "confidence": confidence,
            "hidden_states": outputs.last_hidden_state,
            "attentions": outputs.attentions,
        }
        # Only add embeddings to the dictionary if requested
        if return_embeddings:
            output_dict["embeddings"] = embeddings

        return output_dict
