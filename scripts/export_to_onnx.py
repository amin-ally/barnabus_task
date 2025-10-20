# scripts/export_to_onnx.py
import torch
from pathlib import Path
import yaml
import logging
from src.models.classifier import HateSpeechClassifier
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Exports a trained PyTorch model to the ONNX format for serving.
    """
    parser = argparse.ArgumentParser(description="Export a PyTorch model to ONNX.")
    parser.add_argument(
        "--run_path",
        type=str,
        required=True,
        help="Path to the training run directory containing the 'models' subdir with 'best_model.pt'.",
    )
    args = parser.parse_args()

    run_path = Path(args.run_path)

    logger.info(f"Starting ONNX export process for run: {run_path}")

    # 1. Load configuration
    config_path = Path("config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info("Configuration loaded.")

    # 2. Initialize and load the trained PyTorch model from the specified run
    device = torch.device("cpu")
    model = HateSpeechClassifier(
        model_name=config["model"]["base_model"],
        num_classes=config["model"]["num_classes"],
    )

    checkpoint_path = run_path / "models" / "best_model.pt"
    if not checkpoint_path.exists():
        logger.error(
            f"Checkpoint not found at {checkpoint_path}. Please provide a valid run directory."
        )
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    model.onnx_export_mode = True
    logger.info(
        f"PyTorch model loaded from {checkpoint_path} and set to ONNX export mode."
    )

    # 3. Create a dummy input for tracing
    max_length = config["data"]["max_length"]
    dummy_input_ids = torch.randint(0, 1000, (1, max_length), dtype=torch.long)
    dummy_attention_mask = torch.ones(1, max_length, dtype=torch.long)
    dummy_inputs = (dummy_input_ids, dummy_attention_mask)

    # 4. Define output path and I/O names for the ONNX model
    serving_model_dir = Path(config["serving_paths"]["models"])
    serving_model_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    onnx_output_path = serving_model_dir / "model.onnx"

    input_names = ["input_ids", "attention_mask"]
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
        logger.info(f"✅ Model successfully exported to {onnx_output_path}")
    except Exception as e:
        logger.error(f"❌ ONNX export failed: {e}")


if __name__ == "__main__":
    main()
