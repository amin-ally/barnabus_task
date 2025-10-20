# Multilingual NLP Service for Text Moderation

This project provides a high-performance, containerized API for multilingual text classification, similarity search, and PII (Personally Identifiable Information) masking. The service is designed to identify and categorize text as "safe," "sensitive," or "hateful" in both English and Farsi.

It is built with a robust MLOps pipeline, including automated training, evaluation, ONNX model exportation, and deployment.

## ✨ Features

- **Multilingual Classification**: Classifies text in English and Farsi into safe, sensitive, or hateful categories.
- **High-Performance Inference**: Uses an ONNX-optimized distilbert-base-multilingual-cased model for fast CPU-based inference.
- **Similarity Search**: Finds the most similar messages to a given query from a pre-indexed corpus using a FAISS vector index.
- **PII Masking**: Automatically detects and masks sensitive information like emails, phone numbers, and user mentions before processing.
- **Explainable AI (XAI)**: Provides attention-based token importance scores to help explain model predictions.
- **Containerized Service**: Dockerized for easy, reproducible deployment in any environment.
- **Robust API**: Built with FastAPI, including data validation, rate limiting, and clear documentation.
- **End-to-End MLOps**: A Makefile provides a complete workflow for dependency setup, training, testing, and deployment.

## 🛠️ Technology Stack

- **Backend**: FastAPI, Uvicorn
- **ML/NLP**: PyTorch, Hugging Face Transformers
- **Inference**: ONNX Runtime
- **Vector Search**: Faiss (from Facebook AI)
- **Containerization**: Docker
- **Tooling**: scikit-learn, numpy, pandas

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- make
- Docker (for running the containerized service)

### 1. Installation

Clone the repository and install the required dependencies using the Makefile.

```bash
git clone <your-repo-url>
cd <your-repo-name>
make setup
```

### 2. Data Preparation

Run the data processing and balancing script. This will generate train.csv, val.csv, and test.csv in the data/processed/ directory.

```bash
make prepare-data
```

### 3. Model Training

Train the text classifier. This command will:

- Fine-tune the multilingual model on the processed data.
- Evaluate the model on the test set and save performance metrics.
- Generate vector embeddings for all data splits.
- Save all artifacts (model, metrics, embeddings) into a new timestamped directory inside training_runs/.

```bash
make train
```

After training is complete, you will see a new directory, for example: `training_runs/run_2025-10-20_20-51-31_distilbert-base-multilingual-cased`.

### 4. Deploy the Model for Serving

This crucial step converts the trained PyTorch model to the ONNX format and copies the necessary artifacts to the serving directories defined in `config.yaml`.

You must provide the path to the run directory created in the previous step.

```bash
# Replace with your actual run path
make deploy RUN_PATH=training_runs/run_2025-10-20_20-51-31_distilbert-base-multilingual-cased
```

### 5. Run the API

#### A) Running Locally

Start the FastAPI server using Uvicorn. The server will run on http://localhost:8000.

```bash
make serve
```

#### B) Running with Docker

This project is designed to be deployed as a Docker container. The workflow separates training from deployment for clarity and efficiency.
First, run the training pipeline locally to produce your model artifacts.
Next, use the make deploy command to convert your trained model to the high-performance ONNX format and stage it for the API. You must provide the path to the training run you just created.
Now, build the Docker image. This command packages your FastAPI application along with the ONNX model and vector embeddings you just staged.
```bash
# 1. Build the Docker image
docker build -t multilingual-nlp-service .
```
Run the container to start the service. The API will be available at http://localhost:8000.
```bash
# 2. Run the container
docker run -p 8000:8000 --rm multilingual-nlp-service
```

## ⚙️ API Usage

The API provides the following endpoints.

| Endpoint       | Method | Description                                   |
|----------------|--------|-----------------------------------------------|
| /health       | GET    | Checks the health of the service (model loaded, etc.). |
| /classify     | POST   | Classifies a single piece of text.            |
| /batch_classify | POST | Classifies a batch of up to 50 texts.         |
| /similar      | GET    | Finds K similar messages for a given text query. |

### Examples with curl

#### Text Classification

```bash
curl -X 'POST' \
  'http://localhost:8000/classify' \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "این یک پیام کاملا عادی و بدون مشکل است"
  }'
```

Response:

```json
{
  "original_text": "این یک پیام کاملا عادی و بدون مشکل است",
  "masked_text": "این یک پیام کاملا عادی و بدون مشکل است",
  "language": "fa",
  "label": "safe",
  "confidence": 0.987,
  "probabilities": { "safe": 0.987, "sensitive": 0.01, "hateful": 0.003 },
  "pii_detected": [],
  "explanation": [ /* ... token scores ... */ ]
}
```

#### Similarity Search

```bash
curl -X 'GET' "http://localhost:8000/similar?text=what%20is%20your%20worst%20service&k=3"
```

Response:

```json
{
  "masked_query": "what is your worst service",
  "results": [
    {
      "masked_text": "This is the worst service I have ever experienced.",
      "label": "hateful",
      "similarity": 0.9543
    },
    {
      "masked_text": "I absolutely despise this kind of behavior from people.",
      "label": "hateful",
      "similarity": 0.9211
    },
    {
      "masked_text": "از این شرکت و خدماتش متنفرم",
      "label": "hateful",
      "similarity": 0.8976
    }
  ],
  "pii_detected": []
}
```

## 🧪 Testing and Evaluation

The project includes a suite of tests and benchmarks that can be run with a single command:

```bash
make test
```

This will execute:

- Unit Tests (pytest): Verifies the functionality of the API and data preprocessor.
- Latency Benchmark (scripts/benchmark_latency.py): Measures the p50, p95, and p99 latencies and throughput (RPS) of the /classify endpoint.
- Retrieval Quality Evaluation (scripts/eval_retrieval_ndcg.py): Calculates the nDCG@10 score to evaluate the quality of the similarity search results.

## 📂 Project Structure

```
.
├── api/                # FastAPI application logic
├── Docker/             # Dockerfile for containerization
├── scripts/            # Helper scripts for evaluation, benchmarking, etc.
├── src/                # Core ML model and trainer code
│   ├── models/
├── tests/              # Unit and integration tests
├── training_runs/      # Output directory for all training experiments
├── .dockerignore       # (Recommended) Files to exclude from Docker image
├── config.yaml         # Central configuration file
├── Makefile            # Makefile for automating common tasks
└── train.py            # Main training script
```