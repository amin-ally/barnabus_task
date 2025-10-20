# File: api/main.py (Refactored)

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
import onnxruntime
import torch
import yaml
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from scipy.special import softmax
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from transformers import AutoTokenizer

# Assuming this custom preprocessor exists and is accessible
from data.preprocessor import MultilingualTextPreprocessor

# =============================================================================
# 1. Logging and Application Setup
# =============================================================================

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Multilingual NLP Service",
    description="Classifies text, finds similar messages, and masks PII.",
    version="1.0.1",  # Version bumped to reflect refactoring
)

# Add middlewares (CORS, Rate Limiting)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# =============================================================================
# 2. Pydantic Models for API Data Contracts (Unchanged as requested)
# =============================================================================
class TextRequest(BaseModel):
    text: str = Field(..., description="Text to classify", min_length=1, max_length=512)


class BatchTextRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        description="List of texts to classify",
        min_items=1,
        max_items=50,
    )


class ExplanationFeature(BaseModel):
    token: str
    score: float


class ClassificationResponse(BaseModel):
    original_text: str
    masked_text: str
    language: str
    label: str
    confidence: float
    probabilities: Dict[str, float]
    pii_detected: List[str]
    explanation: Optional[List[ExplanationFeature]] = None


class BatchClassificationResponse(BaseModel):
    results: List[ClassificationResponse]


class SimilarMessage(BaseModel):
    masked_text: str
    label: str
    similarity: float


class SimilarResponse(BaseModel):
    masked_query: str
    results: List[SimilarMessage]
    pii_detected: List[str]


# =============================================================================
# 3. Configuration Management
# =============================================================================
class Settings:
    """Loads and holds application configuration from a YAML file."""

    def __init__(self, config_path: str = "config.yaml"):
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"{config_path} not found!")
        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)

    @property
    def model_name(self) -> str:
        return self.config["model"]["base_model"]

    @property
    def onnx_model_path(self) -> Path:
        return Path(self.config["serving_paths"]["models"]) / "model.onnx"

    @property
    def embeddings_path(self) -> Path:
        return Path(self.config["serving_paths"]["embeddings"]) / "embeddings.npz"

    @property
    def max_input_length(self) -> int:
        return self.config["api"]["max_input_length"]

    @property
    def label_map(self) -> Dict[int, str]:
        return {0: "safe", 1: "sensitive", 2: "hateful"}


# =============================================================================
# 4. Core Service for NLP Tasks
# =============================================================================
class NLPService:
    """Encapsulates all ML models and logic for classification and similarity."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load all assets
        self.tokenizer = self._load_tokenizer()
        self.onnx_session = self._load_onnx_session()
        self.preprocessor = self._initialize_preprocessor()
        self.faiss_index, self.embeddings_data = self._load_faiss_index()

    def _load_tokenizer(self):
        logger.info(f"Loading tokenizer for '{self.settings.model_name}'...")
        return AutoTokenizer.from_pretrained(self.settings.model_name)

    def _load_onnx_session(self):
        model_path = self.settings.onnx_model_path
        if not model_path.exists():
            logger.error(f"ONNX model not found at {model_path}.")
            raise FileNotFoundError(f"ONNX model not found at {model_path}")

        logger.info(f"Loading ONNX model from {model_path}...")
        # Use configurable providers in the future if needed
        return onnxruntime.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )

    def _initialize_preprocessor(self):
        logger.info("Initializing text preprocessor with PII masking...")
        return MultilingualTextPreprocessor(
            max_length=self.settings.max_input_length,
            enable_pii_masking=True,
        )

    def _load_faiss_index(self):
        embeddings_file = self.settings.embeddings_path
        if not embeddings_file.exists():
            logger.warning(
                f"Embeddings file not found at {embeddings_file}. Similarity search will be unavailable."
            )
            return None, None

        logger.info(
            f"Loading embeddings for similarity search from {embeddings_file}..."
        )
        data = np.load(embeddings_file, allow_pickle=True)
        embeddings = data["embeddings"].astype("float32")

        dimension = embeddings.shape[1]
        faiss_index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        faiss_index.add(embeddings)

        embeddings_data = {
            "texts": data["texts"].tolist(),
            "labels": data["labels"].tolist(),
        }
        logger.info(f"Loaded {faiss_index.ntotal} embeddings into FAISS index.")
        return faiss_index, embeddings_data

    def _generate_explanation(
        self, attentions: np.ndarray, input_ids: np.ndarray
    ) -> Optional[List[ExplanationFeature]]:
        """Generates token-level explanations from attention scores."""
        # Average attention across all heads
        avg_head_attentions = attentions.mean(axis=0)
        # Get attention from [CLS] token to all other tokens
        cls_attentions = avg_head_attentions[0]

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        special_tokens = {
            self.tokenizer.cls_token,
            self.tokenizer.sep_token,
            self.tokenizer.pad_token,
        }

        explanation_data = [
            {"token": token, "score": float(score)}
            for token, score in zip(tokens, cls_attentions)
            if token not in special_tokens
        ]

        if not explanation_data:
            return None

        # Normalize scores for better interpretation
        scores = np.array([item["score"] for item in explanation_data])
        if scores.max() > scores.min():
            normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            normalized_scores = np.zeros_like(scores)

        for item, norm_score in zip(explanation_data, normalized_scores):
            item["score"] = round(norm_score, 4)

        explanation_data.sort(key=lambda x: x["score"], reverse=True)
        return [ExplanationFeature(**item) for item in explanation_data[:10]]

    def process_single(self, text: str) -> ClassificationResponse:
        """Processes a single text for classification and explanation."""
        # 1. Preprocessing
        language = self.preprocessor.detect_language(text)
        cleaned_text = self.preprocessor.clean_text(text, language)
        masked_text, pii_types = self.preprocessor.mask_pii(cleaned_text)

        # 2. Tokenization
        encoding = self.tokenizer(
            masked_text,
            truncation=True,
            padding="max_length",
            max_length=self.preprocessor.max_length,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].cpu().numpy()
        attention_mask = encoding["attention_mask"].cpu().numpy()

        # 3. ONNX Inference
        ort_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        logits, attentions = self.onnx_session.run(["logits", "attentions"], ort_inputs)
        probs = softmax(logits, axis=-1).squeeze()

        # 4. Format Response
        predicted_index = int(probs.argmax())
        label = self.settings.label_map.get(predicted_index, "unknown")
        probabilities = {
            self.settings.label_map.get(i, "unknown"): float(prob)
            for i, prob in enumerate(probs)
        }

        # 5. Explanation
        explanation = self._generate_explanation(
            attentions[0], encoding["input_ids"].squeeze(0).numpy()
        )

        return ClassificationResponse(
            original_text=text,
            masked_text=masked_text,
            language=language,
            label=label,
            confidence=float(probs.max()),
            probabilities=probabilities,
            pii_detected=pii_types,
            explanation=explanation,
        )

    def find_similar(self, text: str, k: int) -> SimilarResponse:
        """Finds k similar messages using Faiss."""
        if self.faiss_index is None:
            raise HTTPException(
                status_code=503, detail="Similarity index is not available."
            )

        # 1. Preprocess and embed the query
        language = self.preprocessor.detect_language(text)
        cleaned_text = self.preprocessor.clean_text(text, language)
        masked_query, pii_detected = self.preprocessor.mask_pii(cleaned_text)

        encoding = self.tokenizer(
            cleaned_text,
            truncation=True,
            padding="max_length",
            max_length=self.preprocessor.max_length,
            return_tensors="pt",
        )
        ort_inputs = {
            "input_ids": encoding["input_ids"].cpu().numpy(),
            "attention_mask": encoding["attention_mask"].cpu().numpy(),
        }
        query_embedding = self.onnx_session.run(["embeddings"], ort_inputs)[0]

        # 2. Perform Faiss search
        distances, indices = self.faiss_index.search(query_embedding, k)

        # 3. Format results
        results = [
            SimilarMessage(
                masked_text=self.embeddings_data["texts"][idx][:150]
                + ("..." if len(self.embeddings_data["texts"][idx]) > 150 else ""),
                label=self.settings.label_map.get(
                    self.embeddings_data["labels"][idx], "unknown"
                ),
                similarity=float(dist),
            )
            for idx, dist in zip(indices[0], distances[0])
        ]

        return SimilarResponse(
            masked_query=masked_query, results=results, pii_detected=pii_detected
        )


# =============================================================================
# 5. FastAPI Application Events and Dependencies
# =============================================================================
@app.on_event("startup")
def startup_event():
    """Initializes and loads all ML models and assets on application startup."""
    logger.info("Starting service initialization...")
    try:
        settings = Settings()
        app.state.nlp_service = NLPService(settings)
        logger.info("Service initialization complete.")
    except Exception as e:
        logger.error(f"FATAL: Service failed to initialize: {e}", exc_info=True)
        # In a real-world scenario, you might want to exit the process
        # if the core service fails to load.
        # exit(1)


def get_nlp_service(request: Request) -> NLPService:
    """Dependency to get the NLPService instance from the app state."""
    service = request.app.state.nlp_service
    if not service:
        raise HTTPException(status_code=503, detail="Service is not available.")
    return service


# =============================================================================
# 6. API Endpoints
# =============================================================================
@app.get("/")
@limiter.limit("1000/minute")
async def root(request: Request):
    """Root endpoint with API documentation."""
    return {
        "message": "Multilingual NLP Service is running.",
        "endpoints": {
            "/classify": "POST a single text for classification.",
            "/batch_classify": "POST a list of texts for batch classification.",
            "/similar": "GET similar messages for a given text.",
            "/health": "GET the health status of the service.",
        },
    }


@app.get("/health")
@limiter.limit("1000/minute")
async def health_check(
    request: Request, service: NLPService = Depends(get_nlp_service)
):
    """Health check endpoint to verify service status."""
    index_loaded = service.faiss_index is not None and service.faiss_index.ntotal > 0
    return {
        "status": "healthy",
        "model_loaded": service.onnx_session is not None,
        "index_loaded": index_loaded,
    }


@app.post("/classify", response_model=ClassificationResponse)
@limiter.limit("1000/minute")
def classify_text(
    text_request: TextRequest,
    request: Request,
    service: NLPService = Depends(get_nlp_service),
):
    """
    Classifies input text after cleaning and PII masking.
    Returns the classification, confidence, and token-level explanations.
    """
    start_time = time.perf_counter()
    try:
        result = service.process_single(text_request.text)
        total_time = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"Successfully classified text in {total_time:.2f}ms. "
            f"Input: '{text_request.text[:50]}...'"
        )
        return result
    except Exception as e:
        logger.error(f"Classification error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/batch_classify", response_model=BatchClassificationResponse)
@limiter.limit("100/minute")  # Adjusted limit for a heavier endpoint
def batch_classify_texts(
    batch_request: BatchTextRequest,
    request: Request,
    service: NLPService = Depends(get_nlp_service),
):
    """
    Classifies a batch of texts.
    This endpoint processes each item individually; for true batch performance,
    the NLPService would need a dedicated batch processing method.
    """
    try:
        logger.info(
            f"Processing batch classification request with {len(batch_request.texts)} items."
        )
        # Note: This is a simple loop. For higher throughput, `NLPService`
        # could be enhanced with a method that performs true batch tokenization
        # and inference.
        results = [service.process_single(text) for text in batch_request.texts]
        return BatchClassificationResponse(results=results)
    except Exception as e:
        logger.error(f"Batch classification error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/similar", response_model=SimilarResponse)
@limiter.limit("500/minute")  # Adjusted limit
def find_similar_messages(
    request: Request,
    text: str = Query(
        ..., description="Text to find similar messages for", max_length=512
    ),
    k: int = Query(5, ge=1, le=20, description="Number of results to return"),
    service: NLPService = Depends(get_nlp_service),
):
    """Finds and returns k similar messages from the indexed data."""
    try:
        logger.info(f"Finding {k} similar messages for: '{text[:50]}...'")
        return service.find_similar(text, k)
    except Exception as e:
        logger.error(f"Error finding similar messages: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
