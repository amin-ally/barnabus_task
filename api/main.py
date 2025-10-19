# File: app/main.py

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import torch
import numpy as np
import faiss
from pathlib import Path
import yaml
import logging

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from src.models.classifier import HateSpeechClassifier
from data.preprocessor import MultilingualTextPreprocessor

# Setup structured logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Multilingual NLP Service",
    description="Classifies text, finds similar messages, and masks PII.",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting setup
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# --- Pydantic Models ---
class TextRequest(BaseModel):
    text: str = Field(..., description="Text to classify", min_length=1, max_length=512)


# Model for batch requests
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
    # dd an optional field for the explanation
    explanation: Optional[List[ExplanationFeature]] = None


# NEW: Model for batch responses
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


# --- Global variables and constants ---
model: Optional[HateSpeechClassifier] = None
preprocessor: Optional[MultilingualTextPreprocessor] = None
faiss_index: Optional[faiss.Index] = None
embeddings_data: Optional[Dict] = None
device: Optional[torch.device] = None
LABEL_MAP = {0: "safe", 1: "sensitive", 2: "hateful"}


@app.on_event("startup")
def load_model():
    """Load model and embeddings on startup"""
    global model, preprocessor, faiss_index, embeddings_data, device

    logger.info("Starting service initialization...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    config_path = Path("config.yaml")
    if not config_path.exists():
        logger.error("config.yaml not found! Cannot start the service.")
        raise FileNotFoundError("config.yaml not found!")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Loading multilingual classification model...")
    model = HateSpeechClassifier(
        model_name=config["model"]["base_model"],
        num_classes=config["model"]["num_classes"],
    )

    checkpoint_path = Path(config["paths"]["models"]) / "best_model.pt"
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        logger.info("Successfully loaded model weights.")
    else:
        logger.warning(
            f"Model checkpoint not found at {checkpoint_path}. Using pre-trained weights only."
        )

    model.to(device)
    model.eval()

    logger.info("Initializing text preprocessor with PII masking...")
    preprocessor = MultilingualTextPreprocessor(
        max_length=config["api"]["max_input_length"],
        enable_pii_masking=True,
    )

    logger.info("Loading embeddings for similarity search...")
    embeddings_file = Path(config["paths"]["embeddings"]) / "embeddings.npz"
    if embeddings_file.exists():
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
    else:
        logger.warning(
            f"Embeddings file not found at {embeddings_file}. Similarity search will be unavailable."
        )

    logger.info("Service initialization complete.")


@app.get("/")
@limiter.limit("100/minute")
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
@limiter.limit("100/minute")
async def health_check(request: Request):
    """Health check endpoint to verify service status."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "index_loaded": faiss_index is not None and faiss_index.ntotal > 0,
    }


@app.post("/classify", response_model=ClassificationResponse)
@limiter.limit("60/minute")
async def classify_text(text_request: TextRequest, request: Request):
    """
    Classifies input text after cleaning and PII masking.
    Optionally returns top tokens that influenced the decision.
    """
    if not model or not preprocessor:
        raise HTTPException(status_code=503, detail="Model or preprocessor not loaded.")

    try:
        # Preprocess the text
        language = preprocessor.detect_language(text_request.text)
        cleaned_text = preprocessor.clean_text(text_request.text, language)
        masked_text, pii_types = preprocessor.mask_pii(cleaned_text)

        logger.info(f"Classifying text (lang: {language}): {text_request.text[:50]}...")

        # Tokenize for model input
        encoding = model.tokenizer(
            masked_text,
            truncation=True,
            padding="max_length",
            max_length=preprocessor.max_length,
            return_tensors="pt",
        ).to(device)

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        # Get model prediction
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

        # Format the response
        confidence = float(probs.max())
        predicted_index = int(probs.argmax())
        label = LABEL_MAP.get(predicted_index, "unknown")

        probabilities = {
            LABEL_MAP.get(i, "unknown"): float(prob) for i, prob in enumerate(probs)
        }

        # --- EXPLAINABILITY LOGIC ---
        explanation = None
        # 1. Get attentions from the last layer
        # Shape: [batch_size, num_heads, seq_len, seq_len]
        last_layer_attentions = outputs["attentions"][-1].squeeze(0)  # Remove batch dim

        # 2. Average attention across all heads
        # Shape: [seq_len, seq_len]
        avg_head_attentions = last_layer_attentions.mean(dim=0)

        # 3. Get attention from the [CLS] token to all other tokens
        # The [CLS] token's representation is used for classification.
        # Its attention to other tokens indicates their importance.
        cls_attentions = avg_head_attentions[0, :]  # Index 0 is the [CLS] token

        # 4. Get the tokens
        tokens = model.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
        special_tokens = [
            tok
            for tok in [
                model.tokenizer.cls_token,
                model.tokenizer.bos_token,
                model.tokenizer.sep_token,
                model.tokenizer.pad_token,
            ]
            if tok is not None
        ]
        # 5. Combine tokens and scores, filtering out special tokens
        explanation_data = []
        for token, score in zip(tokens, cls_attentions.cpu().numpy()):
            if token not in special_tokens:
                explanation_data.append({"token": token, "score": float(score)})

        # Process explanations only if we have valid tokens
        if explanation_data:
            # 6. Normalize scores to be more interpretable (e.g., min-max scaling to 0-1)
            scores = np.array([item["score"] for item in explanation_data])
            if scores.max() > scores.min():
                normalized_scores = (scores - scores.min()) / (
                    scores.max() - scores.min()
                )
            else:
                normalized_scores = np.zeros_like(scores)

            for item, norm_score in zip(explanation_data, normalized_scores):
                item["score"] = round(norm_score, 4)

            # 7. Sort by score and take top-k (e.g., top 10)
            explanation_data.sort(key=lambda x: x["score"], reverse=True)
            explanation = [ExplanationFeature(**item) for item in explanation_data[:10]]

        return ClassificationResponse(
            original_text=text_request.text,
            masked_text=masked_text,
            language=language,
            label=label,
            confidence=confidence,
            probabilities=probabilities,
            pii_detected=pii_types,
            explanation=explanation,
        )
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Classification error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# BATCH ENDPOINT WITH EXPLANATIONS
@app.post("/batch_classify", response_model=BatchClassificationResponse)
@limiter.limit("10/minute")  # Lower rate limit for more intensive batch endpoint
async def batch_classify_texts(batch_request: BatchTextRequest, request: Request):
    """
    Classifies a batch of texts after cleaning and PII masking.
    Also returns token-level explanations for each text.
    """
    if not model or not preprocessor:
        raise HTTPException(status_code=503, detail="Model or preprocessor not loaded.")

    try:
        texts = batch_request.texts
        logger.info(f"Processing batch classification request with {len(texts)} items.")

        # 1. Preprocess all texts
        preprocessed_data = []
        masked_texts_for_model = []
        for text in texts:
            language = preprocessor.detect_language(text)
            cleaned_text = preprocessor.clean_text(text, language)
            masked_text, pii_types = preprocessor.mask_pii(cleaned_text)

            preprocessed_data.append(
                {
                    "original_text": text,
                    "masked_text": masked_text,
                    "language": language,
                    "pii_detected": pii_types,
                }
            )
            masked_texts_for_model.append(masked_text)

        # 2. Tokenize the entire batch at once
        encoding = model.tokenizer(
            masked_texts_for_model,
            truncation=True,
            padding=True,  # Pad to the longest sequence in the batch
            max_length=preprocessor.max_length,
            return_tensors="pt",
        ).to(device)

        # 3. Get model predictions for the entire batch
        with torch.no_grad():
            outputs = model(encoding["input_ids"], encoding["attention_mask"])
            logits = outputs["logits"]
            batch_probs = torch.softmax(logits, dim=-1).cpu().numpy()
            # Get attentions for the whole batch
            last_layer_batch_attentions = outputs["attentions"][-1]

        # 4. Format the response for each item in the batch
        results = []
        for i, probs in enumerate(batch_probs):
            confidence = float(probs.max())
            predicted_index = int(probs.argmax())
            label = LABEL_MAP.get(predicted_index, "unknown")

            probabilities = {
                LABEL_MAP.get(j, "unknown"): float(prob) for j, prob in enumerate(probs)
            }

            # --- START: EXPLAINABILITY LOGIC FOR BATCH ITEM ---
            explanation = None
            item_attentions = last_layer_batch_attentions[
                i
            ]  # Shape: [num_heads, seq_len, seq_len]
            item_input_ids = encoding["input_ids"][i]  # Shape: [seq_len]

            avg_head_attentions = item_attentions.mean(dim=0)
            cls_attentions = avg_head_attentions[0, :]
            tokens = model.tokenizer.convert_ids_to_tokens(item_input_ids)

            explanation_data = []
            for token, score in zip(tokens, cls_attentions.cpu().numpy()):
                if token not in [
                    model.tokenizer.cls_token,
                    model.tokenizer.sep_token,
                    model.tokenizer.pad_token,
                ]:
                    explanation_data.append({"token": token, "score": float(score)})

            if explanation_data:
                scores = np.array([item["score"] for item in explanation_data])
                if scores.max() > scores.min():
                    normalized_scores = (scores - scores.min()) / (
                        scores.max() - scores.min()
                    )
                else:
                    normalized_scores = np.zeros_like(scores)

                for item, norm_score in zip(explanation_data, normalized_scores):
                    item["score"] = round(norm_score, 4)

                explanation_data.sort(key=lambda x: x["score"], reverse=True)
                explanation = [
                    ExplanationFeature(**item) for item in explanation_data[:10]
                ]
            # --- END: EXPLAINABILITY LOGIC ---

            item_data = preprocessed_data[i]
            results.append(
                ClassificationResponse(
                    original_text=item_data["original_text"],
                    masked_text=item_data["masked_text"],
                    language=item_data["language"],
                    label=label,
                    confidence=confidence,
                    probabilities=probabilities,
                    pii_detected=item_data["pii_detected"],
                    explanation=explanation,  # Add explanation to response
                )
            )

        return BatchClassificationResponse(results=results)

    except ValueError as e:
        logger.error(f"Validation error in batch processing: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Batch classification error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/similar", response_model=SimilarResponse)
@limiter.limit("60/minute")
async def find_similar_messages(
    request: Request,
    text: str = Query(
        ..., description="Text to find similar messages for", max_length=512
    ),
    k: int = Query(5, ge=1, le=20, description="Number of results to return"),
):
    """Finds and returns k similar messages from the indexed data, masking PII."""
    if faiss_index is None or not model or not preprocessor:
        raise HTTPException(
            status_code=503,
            detail="Similarity index or supporting models are not available.",
        )

    try:
        # Preprocess the query text (clean, mask PII, etc.)
        language = preprocessor.detect_language(text)
        cleaned_text = preprocessor.clean_text(text, language)
        masked_query, pii_detected = preprocessor.mask_pii(cleaned_text)

        log_extra = {
            "client_ip": request.client.host,
            "language": language,
            "k": k,
            "pii_detected": pii_detected,
        }
        logger.info("Similar messages request received", extra=log_extra)

        # Tokenize and create embedding for the query
        encoding = model.tokenizer(
            cleaned_text,  # Use cleaned, unmasked text for better semantic embeddings
            truncation=True,
            padding="max_length",
            max_length=preprocessor.max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(
                encoding["input_ids"],
                encoding["attention_mask"],
                return_embeddings=True,
            )

        query_embedding = outputs["embeddings"].cpu().numpy()

        # Search the FAISS index
        distances, indices = faiss_index.search(query_embedding, k)

        # Format results
        results = []
        for i in range(k):
            idx = indices[0][i]
            dist = distances[0][i]

            original_text = embeddings_data["texts"][idx]
            masked_text, _ = preprocessor.mask_pii(original_text)

            # Truncate for snippet view
            if len(masked_text) > 150:
                masked_text = masked_text[:150] + "..."

            results.append(
                SimilarMessage(
                    masked_text=masked_text,
                    label=LABEL_MAP.get(embeddings_data["labels"][idx], "unknown"),
                    similarity=float(dist),
                )
            )

        return SimilarResponse(
            masked_query=masked_query,
            results=results,
            pii_detected=pii_detected,
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error finding similar messages: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
