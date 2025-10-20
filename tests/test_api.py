# File: tests/test_api.py

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add the parent directory to the path to import the API module
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.main import app, Settings, NLPService
from unittest.mock import Mock, patch, MagicMock
import numpy as np


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = Mock(spec=Settings)
    settings.model_name = "bert-base-multilingual-cased"
    settings.onnx_model_path = Path("models/model.onnx")
    settings.embeddings_path = Path("embeddings/embeddings.npz")
    settings.max_input_length = 512
    settings.label_map = {0: "safe", 1: "sensitive", 2: "hateful"}
    return settings


@pytest.fixture
def mock_nlp_service(mock_settings):
    """Create a mock NLP service for testing."""
    with patch("api.main.NLPService") as MockNLPService:
        service = Mock(spec=NLPService)

        # Mock the process_single method
        service.process_single.return_value = {
            "original_text": "test text",
            "masked_text": "test text",
            "language": "en",
            "label": "safe",
            "confidence": 0.95,
            "probabilities": {"safe": 0.95, "sensitive": 0.03, "hateful": 0.02},
            "pii_detected": [],
            "explanation": None,
        }

        # Mock the find_similar method
        service.find_similar.return_value = {
            "masked_query": "test query",
            "results": [
                {"masked_text": "similar text", "label": "safe", "similarity": 0.89}
            ],
            "pii_detected": [],
        }

        # Mock ONNX session and FAISS index
        service.onnx_session = Mock()
        service.faiss_index = Mock()
        service.faiss_index.ntotal = 100

        return service


@pytest.fixture
def client(mock_nlp_service):
    """Create test client with mocked NLP service."""
    # Override the startup event to use our mock service
    app.state.nlp_service = mock_nlp_service

    with TestClient(app) as test_client:
        yield test_client


class TestAPISmoke:
    """Smoke tests for the API endpoints."""

    def test_root_endpoint(self, client):
        """Test the root endpoint returns expected information."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "endpoints" in data
        assert "/classify" in data["endpoints"]
        assert "/similar" in data["endpoints"]
        assert "/health" in data["endpoints"]

    def test_health_endpoint(self, client):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert "index_loaded" in data

    def test_classify_endpoint_valid(self, client):
        """Test classification with valid input."""
        test_payload = {"text": "This is a test message"}
        response = client.post("/classify", json=test_payload)
        assert response.status_code == 200
        data = response.json()

        # Check required fields in response
        assert "original_text" in data
        assert "masked_text" in data
        assert "language" in data
        assert "label" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert "pii_detected" in data

    def test_classify_endpoint_empty_text(self, client):
        """Test classification with empty text."""
        test_payload = {"text": ""}
        response = client.post("/classify", json=test_payload)
        assert response.status_code == 422  # Validation error

    def test_classify_endpoint_missing_field(self, client):
        """Test classification with missing required field."""
        test_payload = {}
        response = client.post("/classify", json=test_payload)
        assert response.status_code == 422  # Validation error

    def test_classify_endpoint_long_text(self, client):
        """Test classification with text exceeding max length."""
        long_text = "word " * 600  # Exceeds 512 word limit
        test_payload = {"text": long_text}
        response = client.post("/classify", json=test_payload)
        assert response.status_code in [
            200,
            422,
        ]  # Either processes truncated or rejects

    def test_batch_classify_endpoint_valid(self, client):
        """Test batch classification with valid input."""
        test_payload = {
            "texts": ["First test message", "Second test message", "Third test message"]
        }
        response = client.post("/batch_classify", json=test_payload)
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 3

    def test_batch_classify_endpoint_empty_list(self, client):
        """Test batch classification with empty list."""
        test_payload = {"texts": []}
        response = client.post("/batch_classify", json=test_payload)
        assert response.status_code == 422  # Validation error

    def test_batch_classify_endpoint_too_many_texts(self, client):
        """Test batch classification with too many texts."""
        test_payload = {"texts": ["text"] * 51}  # Exceeds max_items=50
        response = client.post("/batch_classify", json=test_payload)
        assert response.status_code == 422  # Validation error

    def test_similar_endpoint_valid(self, client):
        """Test similarity search with valid input."""
        response = client.get("/similar", params={"text": "test query", "k": 5})
        assert response.status_code == 200
        data = response.json()
        assert "masked_query" in data
        assert "results" in data
        assert "pii_detected" in data

    def test_similar_endpoint_missing_text(self, client):
        """Test similarity search with missing text parameter."""
        response = client.get("/similar", params={"k": 5})
        assert response.status_code == 422  # Validation error

    def test_similar_endpoint_invalid_k(self, client):
        """Test similarity search with invalid k value."""
        response = client.get("/similar", params={"text": "test", "k": 0})
        assert response.status_code == 422  # k must be >= 1

        response = client.get("/similar", params={"text": "test", "k": 25})
        assert response.status_code == 422  # k must be <= 20

    def test_similar_endpoint_long_text(self, client):
        """Test similarity search with text exceeding max length."""
        long_text = "a" * 513  # Exceeds max_length=512
        response = client.get("/similar", params={"text": long_text, "k": 5})
        assert response.status_code == 422  # Validation error


class TestAPIWithPII:
    """Test API handling of PII data."""

    def test_classify_with_phone(self, client, mock_nlp_service):
        """Test classification masks phone numbers."""
        mock_nlp_service.process_single.return_value = {
            "original_text": "Call me at 555-1234",
            "masked_text": "Call me at [PHONE]",
            "language": "en",
            "label": "safe",
            "confidence": 0.95,
            "probabilities": {"safe": 0.95, "sensitive": 0.03, "hateful": 0.02},
            "pii_detected": ["PHONE"],
            "explanation": None,
        }

        test_payload = {"text": "Call me at 555-1234"}
        response = client.post("/classify", json=test_payload)
        assert response.status_code == 200
        data = response.json()
        assert "[PHONE]" in data["masked_text"]
        assert "PHONE" in data["pii_detected"]


class TestAPIErrorHandling:
    """Test API error handling scenarios."""

    def test_service_unavailable(self, client):
        """Test response when NLP service is unavailable."""
        # Simulate service not being available
        app.state.nlp_service = None

        response = client.post("/classify", json={"text": "test"})
        assert response.status_code == 503
        assert "Service is not available" in response.json()["detail"]

    def test_invalid_json_payload(self, client):
        """Test response with invalid JSON payload."""
        response = client.post(
            "/classify",
            data="invalid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code in [422, 400]  # Bad request or validation error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
