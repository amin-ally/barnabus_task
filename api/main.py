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
import time

# --- NEW/MODIFIED IMPORTS FOR ONNX ---
import onnxruntime
from transformers import AutoTokenizer
from scipy.special import softmax

# --- END ONNX IMPORTS ---

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

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


# --- Pydantic Models (No changes here) ---
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


# --- MODIFIED Global variables for ONNX ---
onnx_session: Optional[onnxruntime.InferenceSession] = None
tokenizer: Optional[AutoTokenizer] = None
preprocessor: Optional[MultilingualTextPreprocessor] = None
faiss_index: Optional[faiss.Index] = None
embeddings_data: Optional[Dict] = None
device: Optional[torch.device] = None
LABEL_MAP = {0: "safe", 1: "sensitive", 2: "hateful"}


@app.on_event("startup")
def load_model():
    """Load ONNX model, tokenizer, and embeddings on startup"""
    global onnx_session, tokenizer, preprocessor, faiss_index, embeddings_data, device

    logger.info("Starting service initialization...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    config_path = Path("config.yaml")
    if not config_path.exists():
        logger.error("config.yaml not found! Cannot start the service.")
        raise FileNotFoundError("config.yaml not found!")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["base_model"])

    logger.info("Loading ONNX model for inference...")
    onnx_model_path = Path(config["paths"]["models"]) / "model.onnx"
    if onnx_model_path.exists():
        # Use CPUExecutionProvider for CPU-based inference
        onnx_session = onnxruntime.InferenceSession(
            str(onnx_model_path), providers=["CPUExecutionProvider"]
        )
        logger.info("Successfully loaded ONNX model.")
    else:
        logger.error(
            f"ONNX model not found at {onnx_model_path}. Please run the export script first."
        )
        raise FileNotFoundError(f"ONNX model not found at {onnx_model_path}")

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
async def health_check(request: Request):
    """Health check endpoint to verify service status."""
    return {
        "status": "healthy",
        "model_loaded": onnx_session is not None,
        "index_loaded": faiss_index is not None and faiss_index.ntotal > 0,
    }


@app.post("/classify", response_model=ClassificationResponse)
@limiter.limit("1000/minute")
async def classify_text(text_request: TextRequest, request: Request):
    """
    Classifies input text after cleaning and PII masking using the ONNX model.
    Optionally returns top tokens that influenced the decision.
    """
    start_total_time = time.perf_counter()

    if not onnx_session or not preprocessor or not tokenizer:
        raise HTTPException(status_code=503, detail="Model or preprocessor not loaded.")

    try:
        # --- 1. Preprocessing ---
        start_preprocess_time = time.perf_counter()
        language = preprocessor.detect_language(text_request.text)
        cleaned_text = preprocessor.clean_text(text_request.text, language)
        masked_text, pii_types = preprocessor.mask_pii(cleaned_text)
        preprocess_time = (time.perf_counter() - start_preprocess_time) * 1000

        logger.info(f"Classifying text (lang: {language}): {text_request.text[:50]}...")

        # --- 2. Tokenization ---
        start_tokenize_time = time.perf_counter()
        encoding = tokenizer(
            masked_text,
            truncation=True,
            padding="max_length",
            max_length=preprocessor.max_length,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        tokenize_time = (time.perf_counter() - start_tokenize_time) * 1000

        # --- 3. ONNX Model Inference ---
        start_inference_time = time.perf_counter()
        ort_inputs = {
            "input_ids": input_ids.cpu().numpy(),
            "attention_mask": attention_mask.cpu().numpy(),
        }
        ort_outs = onnx_session.run(["logits", "attentions"], ort_inputs)
        logits, last_layer_attentions = ort_outs[0], ort_outs[1]
        probs = softmax(logits, axis=-1).squeeze()
        inference_time = (time.perf_counter() - start_inference_time) * 1000

        # --- 4. Response Formatting ---
        confidence = float(probs.max())
        predicted_index = int(probs.argmax())
        label = LABEL_MAP.get(predicted_index, "unknown")
        probabilities = {
            LABEL_MAP.get(i, "unknown"): float(prob) for i, prob in enumerate(probs)
        }

        # --- 5. Explainability Logic (adapted for NumPy) ---
        start_explain_time = time.perf_counter()
        explanation = None

        # Shape changes from (num_heads, seq_len, seq_len) to (seq_len, seq_len).
        avg_head_attentions = last_layer_attentions.mean(axis=0)

        # 2. Get the attention scores from the [CLS] token (the first row) to all other tokens.
        # This is now a 1D array of floats.
        cls_attentions = avg_head_attentions[0][0]

        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
        special_tokens = [
            tok
            for tok in [
                tokenizer.cls_token,
                tokenizer.bos_token,
                tokenizer.sep_token,
                tokenizer.pad_token,
            ]
            if tok is not None
        ]
        explanation_data = []
        for token, score in zip(tokens, cls_attentions):
            if token not in special_tokens:
                print(score)
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
            explanation = [ExplanationFeature(**item) for item in explanation_data[:10]]
        explain_time = (time.perf_counter() - start_explain_time) * 1000

        total_time = (time.perf_counter() - start_total_time) * 1000

        logger.info(
            f"Timing profile for classify endpoint: "
            f"Total={total_time:.2f}ms, "
            f"Preprocess={preprocess_time:.2f}ms, "
            f"Tokenize={tokenize_time:.2f}ms, "
            f"Inference={inference_time:.2f}ms, "
            f"Explainability={explain_time:.2f}ms"
        )

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


@app.post("/batch_classify", response_model=BatchClassificationResponse)
@limiter.limit("1000/minute")
async def batch_classify_texts(batch_request: BatchTextRequest, request: Request):
    """
    Classifies a batch of texts using the ONNX model.
    Also returns token-level explanations for each text.
    """
    if not onnx_session or not preprocessor or not tokenizer:
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

        # 2. Tokenize the entire batch
        encoding = tokenizer(
            masked_texts_for_model,
            truncation=True,
            padding=True,
            max_length=preprocessor.max_length,
            return_tensors="pt",
        )

        # 3. Get ONNX model predictions for the batch
        ort_inputs = {
            "input_ids": encoding["input_ids"].cpu().numpy(),
            "attention_mask": encoding["attention_mask"].cpu().numpy(),
        }
        ort_outs = onnx_session.run(["logits", "attentions"], ort_inputs)
        batch_logits, last_layer_batch_attentions = ort_outs[0], ort_outs[1]
        batch_probs = softmax(batch_logits, axis=-1)

        # 4. Format the response for each item
        results = []
        for i, probs in enumerate(batch_probs):
            confidence = float(probs.max())
            predicted_index = int(probs.argmax())
            label = LABEL_MAP.get(predicted_index, "unknown")

            probabilities = {
                LABEL_MAP.get(j, "unknown"): float(prob) for j, prob in enumerate(probs)
            }

            # --- START: EXPLAINABILITY LOGIC FOR BATCH ITEM (CORRECTED) ---
            explanation = None
            # Select attentions for the current item from the batch
            item_attentions = last_layer_batch_attentions[i]
            item_input_ids = encoding["input_ids"][i]

            # Average attention across all heads (axis=0)
            avg_head_attentions = item_attentions.mean(axis=0)

            # Following your successful fix: get the CLS token's attention scores (the first row)
            cls_attentions = avg_head_attentions[0]

            tokens = tokenizer.convert_ids_to_tokens(item_input_ids)
            special_tokens = [
                tok
                for tok in [
                    tokenizer.cls_token,
                    tokenizer.sep_token,
                    tokenizer.pad_token,
                ]
                if tok is not None
            ]

            explanation_data = []
            if cls_attentions.size > 0:
                for token, score in zip(tokens, cls_attentions):
                    if token not in special_tokens:
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
                    explanation=explanation,
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
@limiter.limit("1000/minute")
async def find_similar_messages(
    request: Request,
    text: str = Query(
        ..., description="Text to find similar messages for", max_length=512
    ),
    k: int = Query(5, ge=1, le=20, description="Number of results to return"),
):
    """Finds and returns k similar messages using embeddings from the ONNX model."""
    if faiss_index is None or not onnx_session or not preprocessor:
        raise HTTPException(
            status_code=503,
            detail="Similarity index or supporting models are not available.",
        )

    try:
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

        encoding = tokenizer(
            cleaned_text,
            truncation=True,
            padding="max_length",
            max_length=preprocessor.max_length,
            return_tensors="pt",
        )

        # --- ONNX Inference for Embeddings ---
        ort_inputs = {
            "input_ids": encoding["input_ids"].cpu().numpy(),
            "attention_mask": encoding["attention_mask"].cpu().numpy(),
        }
        query_embedding = onnx_session.run(["embeddings"], ort_inputs)[0]

        # ================================================================= #
        query_dim = query_embedding.shape[1]
        index_dim = faiss_index.d

        if query_dim != index_dim:
            error_detail = (
                f"Dimension mismatch: The query vector has dimension {query_dim}, "
                f"but the FAISS index was built with dimension {index_dim}. "
                "This indicates that the model running in the API is different from the one "
                "used to generate the embeddings. Please re-run the full training, export, "
                "and serving pipeline with a consistent 'base_model' in config.yaml."
            )
            logger.error(error_detail)
            raise HTTPException(status_code=500, detail=error_detail)
        # ================================================================= #

        distances, indices = faiss_index.search(query_embedding, k)

        results = []
        for i in range(k):
            idx = indices[0][i]
            dist = distances[0][i]

            original_text = embeddings_data["texts"][idx]
            masked_text, _ = preprocessor.mask_pii(original_text)

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
