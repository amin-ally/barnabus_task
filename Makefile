.PHONY: setup train serve test clean

# ==============================================================================
# Project Setup & Dependencies
# ==============================================================================
setup:
	@echo ">> Installing dependencies from requirements.txt..."
	pip install -r requirements.txt
	@echo ">> Setup complete."

# ==============================================================================
# Data & Model Pipeline
# ==============================================================================
# Note: You might want a separate data download/prep script.
# For now, we assume data is placed in `./data` manually.
prepare-data:
	@echo ">> Running data preparation..."
	# This assumes you move the logic from data_loader's __main__ to a script
	 python -m data.data_loader
	@echo "Data preparation is currently manual. Ensure data is in the ./data directory."


train:
	@echo ">> Starting model training..."
	python -m train
	@echo ">> Training complete. Model and embeddings saved to ./models and ./data/embeddings."

# ==============================================================================
# Service & API
# ==============================================================================
serve:
	@echo ">> Starting FastAPI server at http://localhost:8000..."
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# ==============================================================================
# Cleanup
# ==============================================================================
clean:
	@echo ">> Cleaning up Python cache files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	@echo ">> Cleanup complete."


# ==============================================================================
# Deployment
# ==============================================================================
# This command "promotes" a trained model to be used by the serving API.
# It copies the embeddings and exports the PyTorch model to ONNX format in the serving directories.
# Usage: make deploy RUN_PATH=training_runs/run_2025-10-20_15-00-00_distilbert-base-multilingual-cased
deploy:
	@if [ -z "$(RUN_PATH)" ]; then \
		echo "ERROR: Please specify the path to the run directory."; \
		echo "Usage: make deploy RUN_PATH=<path_to_your_run_directory>"; \
		exit 1; \
	fi
	@echo ">> Deploying model from run: $(RUN_PATH)"
	
	# Load serving paths from config.yaml
	@{ \
		SERVING_EMBEDDINGS_DIR=$$(python -c "import yaml; print(yaml.safe_load(open('config.yaml'))['serving_paths']['embeddings'])"); \
		SERVING_MODELS_DIR=$$(python -c "import yaml; print(yaml.safe_load(open('config.yaml'))['serving_paths']['models'])"); \
		\
		echo ">> Creating serving directories if they don't exist..."; \
		mkdir -p $$SERVING_EMBEDDINGS_DIR; \
		mkdir -p $$SERVING_MODELS_DIR; \
		\
		echo ">> Copying embeddings to $$SERVING_EMBEDDINGS_DIR..."; \
		cp "$(RUN_PATH)/embeddings/embeddings.npz" "$$SERVING_EMBEDDINGS_DIR/embeddings.npz"; \
		\
		echo ">> Exporting model to ONNX format at $$SERVING_MODELS_DIR..."; \
		python -m scripts.export_to_onnx --run_path "$(RUN_PATH)"; \
		\
		echo "✅ Deployment complete. You can now start the API with 'make serve'."; \
	}
