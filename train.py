# train.py
import os
import yaml
import torch
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
import json

# Import our modules
from src.models.classifier import HateSpeechClassifier
from src.models.trainer import ModelTrainer, TextDataset

import torch.optim as optim

from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path="config.yaml"):
    """Load configuration from yaml file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_preprocessed_data(data_dir="./data/processed"):
    """
    Load preprocessed train, validation, and test data from CSV files.

    Parameters
    ----------
    data_dir : str
        Directory containing the preprocessed CSV files (train.csv, val.csv, test.csv)

    Returns
    -------
    dict
        Dictionary containing 'train', 'val', and 'test' dataframes
    """
    logger.info("Loading preprocessed data from CSV files...")

    data_path = Path(data_dir)

    # Check if directory exists
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_path.absolute()}\n"
            f"Please run the data preparation script first to generate the processed datasets."
        )

    splits = {}

    # Load each split
    for split_name in ["train", "val", "test"]:
        file_path = data_path / f"{split_name}.csv"

        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {file_path.absolute()}\n"
                f"Please ensure all three files (train.csv, val.csv, test.csv) exist in {data_dir}"
            )

        # Load CSV
        df = pd.read_csv(file_path)

        # Validate required columns
        required_columns = ["text", "label_text", "label", "language", "source"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(
                f"Missing required columns in {split_name}.csv: {missing_columns}\n"
                f"Available columns: {list(df.columns)}"
            )

        splits[split_name] = df

        # Log statistics
        logger.info(f"\n{'='*50}")
        logger.info(f"Loaded {split_name.upper()} split")
        logger.info(f"{'='*50}")
        logger.info(f"Total samples: {len(df):,}")
        logger.info(f"Languages: {df['language'].value_counts().to_dict()}")
        logger.info(f"Labels: {df['label_text'].value_counts().to_dict()}")
        logger.info(f"Sources: {df['source'].value_counts().to_dict()}")

        # Per-language label distribution
        for lang in df["language"].unique():
            lang_df = df[df["language"] == lang]
            logger.info(f"\n{lang.upper()} label distribution:")
            logger.info(f"{lang_df['label_text'].value_counts().to_dict()}")

    # Summary statistics
    logger.info(f"\n{'='*50}")
    logger.info(f"DATA LOADING SUMMARY")
    logger.info(f"{'='*50}")
    total_samples = sum(len(df) for df in splits.values())
    logger.info(f"Total samples across all splits: {total_samples:,}")
    logger.info(
        f"Train: {len(splits['train']):,} ({len(splits['train'])/total_samples*100:.1f}%)"
    )
    logger.info(
        f"Val: {len(splits['val']):,} ({len(splits['val'])/total_samples*100:.1f}%)"
    )
    logger.info(
        f"Test: {len(splits['test']):,} ({len(splits['test'])/total_samples*100:.1f}%)"
    )

    return splits


def create_data_loaders(config, splits):
    """
    Create PyTorch data loaders from preprocessed splits.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    splits : dict
        Dictionary containing 'train', 'val', and 'test' dataframes

    Returns
    -------
    tuple
        (train_loader, val_loader, test_loader)
    """
    logger.info("\nCreating PyTorch data loaders...")

    # Initialize model to get tokenizer
    model = HateSpeechClassifier(
        model_name=config["model"]["base_model"],
        num_classes=config["model"]["num_classes"],
        dropout_rate=config["model"]["dropout_rate"],
    )

    # Create datasets
    train_dataset = TextDataset(
        texts=splits["train"]["text"].tolist(),
        labels=splits["train"]["label"].tolist(),
        tokenizer=model.tokenizer,
        max_length=config["data"]["max_length"],
    )

    val_dataset = TextDataset(
        texts=splits["val"]["text"].tolist(),
        labels=splits["val"]["label"].tolist(),
        tokenizer=model.tokenizer,
        max_length=config["data"]["max_length"],
    )

    test_dataset = TextDataset(
        texts=splits["test"]["text"].tolist(),
        labels=splits["test"]["label"].tolist(),
        tokenizer=model.tokenizer,
        max_length=config["data"]["max_length"],
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"].get("num_workers", 2),
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"].get("num_workers", 2),
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"].get("num_workers", 2),
        pin_memory=True if torch.cuda.is_available() else False,
    )

    logger.info(f"✓ Created train loader: {len(train_loader)} batches")
    logger.info(f"✓ Created val loader: {len(val_loader)} batches")
    logger.info(f"✓ Created test loader: {len(test_loader)} batches")

    return train_loader, val_loader, test_loader, model


def train_model(config, train_loader, val_loader, test_loader, model):
    """Train the classification model"""
    logger.info("\n" + "=" * 60)
    logger.info("STARTING MODEL TRAINING")
    logger.info("=" * 60)

    # Initialize trainer
    trainer = ModelTrainer(
        model=model,
        learning_rate=config["training"]["learning_rate"],
        warmup_steps=config["training"]["warmup_steps"],
    )

    # Setup optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        eps=1e-8,
        weight_decay=config["training"].get("weight_decay", 0.01),
    )

    total_steps = len(train_loader) * config["training"]["num_epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["training"]["warmup_steps"],
        num_training_steps=total_steps,
    )

    # Training loop
    train_losses = []
    val_losses = []
    train_f1s = []
    val_f1s = []
    best_val_f1 = 0
    patience_counter = 0

    for epoch in range(config["training"]["num_epochs"]):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{config['training']['num_epochs']}")
        logger.info(f"{'='*60}")

        # Train
        train_loss, train_f1 = trainer.train_epoch(train_loader, optimizer, scheduler)
        train_losses.append(train_loss)
        train_f1s.append(train_f1)

        # Validate
        val_metrics = trainer.evaluate(val_loader)
        val_losses.append(val_metrics["loss"])
        val_f1s.append(val_metrics["f1"])

        logger.info(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
        logger.info(
            f"Val Loss: {val_metrics['loss']:.4f}, Val F1: {val_metrics['f1']:.4f}"
        )
        logger.info(f"Val Avg Confidence: {val_metrics['avg_confidence']:.4f}")

        # Save best model
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            patience_counter = 0

            # Save model
            model_path = Path(config["paths"]["models"]) / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1": val_metrics["f1"],
                    "config": config,
                },
                model_path,
            )
            logger.info(f"✓ Saved best model with Val F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config["training"]["early_stopping_patience"]:
            logger.info(f"⚠️  Early stopping triggered after {epoch + 1} epochs")
            break

    # Plot training history
    plot_training_history(
        train_losses,
        val_losses,
        train_f1s,
        val_f1s,
        save_path=Path(config["paths"]["models"]) / "training_history.png",
    )

    # Load best model for final evaluation
    logger.info("\n" + "=" * 60)
    logger.info("FINAL EVALUATION ON TEST SET")
    logger.info("=" * 60)

    checkpoint = torch.load(Path(config["paths"]["models"]) / "best_model.pt")
    model.load_state_dict(checkpoint["model_state_dict"])

    # Final test evaluation
    test_metrics = trainer.evaluate(test_loader)

    # Generate classification report
    label_names = ["safe", "sensitive", "hateful"]  # Based on label mapping: 0, 1, 2
    report = classification_report(
        test_metrics["true_labels"],
        test_metrics["predictions"],
        target_names=label_names,
        output_dict=True,
    )

    logger.info("\nClassification Report:")
    print(
        classification_report(
            test_metrics["true_labels"],
            test_metrics["predictions"],
            target_names=label_names,
        )
    )

    # Plot confusion matrix
    plot_confusion_matrix(
        test_metrics["true_labels"],
        test_metrics["predictions"],
        label_names,
        save_path=Path(config["paths"]["models"]) / "confusion_matrix.png",
    )

    # Save metrics
    metrics = {
        "test_f1": float(test_metrics["f1"]),
        "test_loss": float(test_metrics["loss"]),
        "test_confidence": float(test_metrics["avg_confidence"]),
        "classification_report": report,
        "best_val_f1": float(best_val_f1),
        "training_epochs": epoch + 1,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
    }

    metrics_path = Path(config["paths"]["models"]) / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"\n✓ Saved metrics to {metrics_path}")

    return model, trainer, metrics


def plot_training_history(
    train_losses, val_losses, train_f1s, val_f1s, save_path="training_history.png"
):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, "b-o", label="Train Loss", linewidth=2, markersize=6)
    ax1.plot(epochs, val_losses, "r-s", label="Val Loss", linewidth=2, markersize=6)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot F1 scores
    ax2.plot(epochs, train_f1s, "b-o", label="Train F1", linewidth=2, markersize=6)
    ax2.plot(epochs, val_f1s, "r-s", label="Val F1", linewidth=2, markersize=6)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("F1 Score", fontsize=12)
    ax2.set_title("Training and Validation F1 Score", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    logger.info(f"✓ Saved training history plot to {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, labels, save_path="confusion_matrix.png"):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Count"},
        annot_kws={"size": 14},
    )
    plt.title("Confusion Matrix", fontsize=16, fontweight="bold", pad=20)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    logger.info(f"✓ Saved confusion matrix to {save_path}")
    plt.close()


def generate_embeddings(model, splits, config):
    """Generate embeddings for all data"""
    logger.info("\n" + "=" * 60)
    logger.info("GENERATING EMBEDDINGS FOR SIMILARITY SEARCH")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_embeddings = []
    all_texts = []
    all_labels = []
    all_languages = []
    all_sources = []

    for split_name, split_df in splits.items():
        logger.info(f"\nProcessing {split_name} split...")

        dataset = TextDataset(
            texts=split_df["text"].tolist(),
            labels=split_df["label"].tolist(),
            tokenizer=model.tokenizer,
            max_length=config["data"]["max_length"],
        )

        loader = DataLoader(
            dataset,
            batch_size=config["training"].get("embedding_batch_size", 32),
            shuffle=False,
            num_workers=2,
        )

        split_embeddings = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                outputs = model(input_ids, attention_mask, return_embeddings=True)
                split_embeddings.append(outputs["embeddings"].cpu().numpy())

        split_embeddings = np.vstack(split_embeddings)
        all_embeddings.append(split_embeddings)
        all_texts.extend(split_df["text"].tolist())
        all_labels.extend(split_df["label"].tolist())
        all_languages.extend(split_df["language"].tolist())
        all_sources.extend(split_df["source"].tolist())

        logger.info(f"✓ Generated {len(split_embeddings)} embeddings for {split_name}")

    # Combine all embeddings
    all_embeddings = np.vstack(all_embeddings)

    # Save embeddings
    embeddings_path = Path(config["paths"]["embeddings"])
    embeddings_path.mkdir(parents=True, exist_ok=True)

    embeddings_file = embeddings_path / "embeddings.npz"
    np.savez(
        embeddings_file,
        embeddings=all_embeddings,
        texts=all_texts,
        labels=all_labels,
        languages=all_languages,
        sources=all_sources,
    )

    logger.info(f"\n✓ Saved {len(all_embeddings)} embeddings to {embeddings_file}")
    logger.info(f"  Embedding shape: {all_embeddings.shape}")

    # Create metadata file
    metadata = {
        "total_embeddings": len(all_embeddings),
        "embedding_dim": all_embeddings.shape[1],
        "languages": list(set(all_languages)),
        "sources": list(set(all_sources)),
        "label_distribution": {
            str(label): int(count)
            for label, count in pd.Series(all_labels).value_counts().items()
        },
    }

    metadata_path = embeddings_path / "embeddings_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"✓ Saved embedding metadata to {metadata_path}")

    return {
        "embeddings": all_embeddings,
        "texts": all_texts,
        "labels": all_labels,
        "languages": all_languages,
        "sources": all_sources,
    }


def main():
    """Main training pipeline"""
    logger.info("\n" + "=" * 60)
    logger.info("MULTILINGUAL HATE SPEECH CLASSIFICATION - TRAINING PIPELINE")
    logger.info("=" * 60)

    # Load configuration
    config = load_config()

    logger.info(f"\nConfiguration loaded:")
    logger.info(f"  Base model: {config['model']['base_model']}")
    logger.info(f"  Batch size: {config['training']['batch_size']}")
    logger.info(f"  Learning rate: {config['training']['learning_rate']}")
    logger.info(f"  Epochs: {config['training']['num_epochs']}")

    # Create timestamped output directories for this run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(config["paths"]["models"]) / f"run_{timestamp}"
    embeddings_dir = Path(config["paths"]["embeddings"]) / f"run_{timestamp}"

    run_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    # --- Update config paths so subsequent functions write into this run ---
    config["paths"]["models"] = str(run_dir)
    config["paths"]["embeddings"] = str(embeddings_dir)

    logger.info(f"  Created run directory: {run_dir}")
    logger.info(f"  Created embeddings directory: {embeddings_dir}")

    # Load preprocessed data
    splits = load_preprocessed_data(data_dir="./data/processed")

    # Create data loaders
    train_loader, val_loader, test_loader, model = create_data_loaders(config, splits)

    # Train model
    model, trainer, metrics = train_model(
        config, train_loader, val_loader, test_loader, model
    )

    # Generate embeddings
    embeddings_data = generate_embeddings(model, splits, config)

    # Save run summary to a small info file for easy tracking
    run_info = {
        "timestamp": timestamp,
        "base_model": config["model"]["base_model"],
        "num_epochs": metrics["training_epochs"],
        "best_val_f1": metrics["best_val_f1"],
        "test_f1": metrics["test_f1"],
        "model_path": str(run_dir / "best_model.pt"),
        "metrics_path": str(run_dir / "metrics.json"),
        "embeddings_path": str(embeddings_dir / "embeddings.npz"),
    }

    with open(run_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    logger.info(f"Final Test F1 Score: {metrics['test_f1']:.4f}")
    logger.info(f"Best Validation F1 Score: {metrics['best_val_f1']:.4f}")
    logger.info(f"Total Training Epochs: {metrics['training_epochs']}")
    logger.info(f"Model saved to: {run_dir}")
    logger.info(f"Embeddings saved to: {embeddings_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
