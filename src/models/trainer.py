# src/models/trainer.py
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, classification_report
import numpy as np
from tqdm import tqdm
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """Custom dataset for text classification"""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
        }


class ModelTrainer:
    """Trainer for the classification model"""

    def __init__(
        self,
        model,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 2e-5,
        warmup_steps: int = 500,
    ):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.best_val_f1 = 0

    def train_epoch(self, dataloader, optimizer, scheduler):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        predictions = []
        true_labels = []

        for batch in tqdm(dataloader, desc="Training"):
            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            # Forward pass
            outputs = self.model(input_ids, attention_mask)
            loss = torch.nn.functional.cross_entropy(outputs["logits"], labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            predictions.extend(outputs["predictions"].detach().cpu().numpy())
            true_labels.extend(labels.detach().cpu().numpy())

        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        f1 = f1_score(true_labels, predictions, average="macro")

        return avg_loss, f1

    def evaluate(self, dataloader):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        confidences = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                loss = torch.nn.functional.cross_entropy(outputs["logits"], labels)

                total_loss += loss.item()
                predictions.extend(outputs["predictions"].detach().cpu().numpy())
                true_labels.extend(labels.detach().cpu().numpy())
                confidences.extend(outputs["confidence"].cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        f1 = f1_score(true_labels, predictions, average="macro")
        avg_confidence = np.mean(confidences)

        return {
            "loss": avg_loss,
            "f1": f1,
            "avg_confidence": avg_confidence,
            "predictions": predictions,
            "true_labels": true_labels,
        }
