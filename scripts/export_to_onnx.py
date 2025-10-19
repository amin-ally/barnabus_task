# scripts/export_to_onnx.py
import torch
from pathlib import Path
import yaml
import logging
from src.models.classifier import HateSpeechClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Exports the fine-tuned PyTorch model to the ONNX format for accelerated inference.
    """
    logger.info("Starting ONNX export process...")

    # 1. Load configuration
    config_path = Path("config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info("Configuration loaded.")

    # 2. Initialize and load the trained PyTorch model
    device = torch.device("cpu")  # Export must be done on CPU
    model = HateSpeechClassifier(
        model_name=config["model"]["base_model"],
        num_classes=config["model"]["num_classes"],
    )

    checkpoint_path = Path(config["paths"]["models"]) / "best_model.pt"
    if not checkpoint_path.exists():
        logger.error(
            f"Checkpoint not found at {checkpoint_path}. Please train the model first."
        )
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    model.onnx_export_mode = True  # Set the special export flag
    logger.info("PyTorch model loaded and set to ONNX export mode.")

    # 3. Create a dummy input for tracing the model graph
    max_length = config["data"]["max_length"]
    # The first dimension is batch_size, the second is sequence_length
    dummy_input_ids = torch.randint(0, 1000, (1, max_length), dtype=torch.long)
    dummy_attention_mask = torch.ones(1, max_length, dtype=torch.long)
    dummy_inputs = (dummy_input_ids, dummy_attention_mask)
    logger.info(f"Dummy input created with shape (1, {max_length}).")

    # 4. Define output path and I/O names for the ONNX model
    onnx_output_path = Path(config["paths"]["models"]) / "model.onnx"
    input_names = ["input_ids", "attention_mask"]
    # These names correspond to the tuple returned by the forward method
    output_names = ["logits", "attentions", "embeddings"]
    logger.info(f"ONNX model will be saved to: {onnx_output_path}")

    # 5. Export the model
    try:
        torch.onnx.export(
            model,
            args=dummy_inputs,
            f=str(onnx_output_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size"},
                "attentions": {0: "batch_size"},
                "embeddings": {0: "batch_size"},
            },
            opset_version=11,
            do_constant_folding=True,
        )
        logger.info("✅ Model successfully exported to ONNX format!")
    except Exception as e:
        logger.error(f"❌ ONNX export failed: {e}")


if __name__ == "__main__":
    main()  # scripts/export_to_onnx.py
import torch
from pathlib import Path
import yaml
import logging
from src.models.classifier import HateSpeechClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Exports the fine-tuned PyTorch model to the ONNX format for accelerated inference.
    """
    logger.info("Starting ONNX export process...")

    # 1. Load configuration
    config_path = Path("config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info("Configuration loaded.")

    # 2. Initialize and load the trained PyTorch model
    device = torch.device("cpu")  # Export must be done on CPU
    model = HateSpeechClassifier(
        model_name=config["model"]["base_model"],
        num_classes=config["model"]["num_classes"],
    )

    checkpoint_path = Path(config["paths"]["models"]) / "best_model.pt"
    if not checkpoint_path.exists():
        logger.error(
            f"Checkpoint not found at {checkpoint_path}. Please train the model first."
        )
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    model.onnx_export_mode = True  # Set the special export flag
    logger.info("PyTorch model loaded and set to ONNX export mode.")

    # 3. Create a dummy input for tracing the model graph
    max_length = config["data"]["max_length"]
    # The first dimension is batch_size, the second is sequence_length
    dummy_input_ids = torch.randint(0, 1000, (1, max_length), dtype=torch.long)
    dummy_attention_mask = torch.ones(1, max_length, dtype=torch.long)
    dummy_inputs = (dummy_input_ids, dummy_attention_mask)
    logger.info(f"Dummy input created with shape (1, {max_length}).")

    # 4. Define output path and I/O names for the ONNX model
    onnx_output_path = Path(config["paths"]["models"]) / "model.onnx"
    input_names = ["input_ids", "attention_mask"]
    # These names correspond to the tuple returned by the forward method
    output_names = ["logits", "attentions", "embeddings"]
    logger.info(f"ONNX model will be saved to: {onnx_output_path}")

    # 5. Export the model
    try:
        torch.onnx.export(
            model,
            args=dummy_inputs,
            f=str(onnx_output_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size"},
                "attentions": {0: "batch_size"},
                "embeddings": {0: "batch_size"},
            },
            opset_version=11,
            do_constant_folding=True,
        )
        logger.info("✅ Model successfully exported to ONNX format!")
    except Exception as e:
        logger.error(f"❌ ONNX export failed: {e}")


if __name__ == "__main__":
    main()
