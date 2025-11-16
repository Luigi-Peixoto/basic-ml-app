import os
import sys
import pytest
from unittest.mock import MagicMock
from dotenv import load_dotenv

# Add the project root to the path to allow importing from 'app' and 'intent_classifier'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from fastapi import HTTPException
from app.app import app
from intent_classifier import IntentClassifier, Config

# --- Fixtures ---

@pytest.fixture(scope="function", autouse=True)
def mock_app_dependencies(monkeypatch, request):
    """
    Auto-used fixture to mock external dependencies for unit tests.
    For integration tests, it only mocks the database collection.
    """
    mock_collection = MagicMock()
    # Mock the factory function to ensure the app uses our mock collection
    monkeypatch.setattr("db.engine.get_mongo_collection", lambda name: mock_collection)

    if "integration" in request.node.keywords:
        yield mock_collection, None, None
        return

    # --- Full Mocks for Unit Tests ---
    mock_model = MagicMock(spec=IntentClassifier)
    mock_model.predict.return_value = ("mock_intent", {"mock_intent": 0.9, "other": 0.1})
    
    # Mock the function that loads models during app startup
    mock_load = MagicMock(return_value={"mock-model": mock_model})
    monkeypatch.setattr("app.services.load_all_classifiers", mock_load)

    mock_verify_token = MagicMock(return_value="mock_prod_user")
    monkeypatch.setattr("db.auth.verify_token", mock_verify_token)

    yield mock_collection, mock_model, mock_verify_token

@pytest.fixture(scope="function")
def client():
    """Provides a TestClient for making in-memory requests to the app."""
    with TestClient(app) as test_client:
        yield test_client

# --- Unit Tests ---

def test_predict_dev_mode(client, monkeypatch, mock_app_dependencies):
    """Tests POST /predict in 'dev' mode, which should bypass authentication."""
    monkeypatch.setattr("db.auth.ENV", "dev")
    mock_collection, mock_model, mock_verify_token = mock_app_dependencies
    
    response = client.post("/predict", params={"text": "hello dev mode"})
    
    assert response.status_code == 200
    data = response.json()
    assert data["owner"] == "dev_user"
    assert "mock-model" in data["predictions"]
    
    mock_verify_token.assert_not_called()
    mock_model.predict.assert_called_once_with("hello dev mode")
    mock_collection.insert_one.assert_called_once()

def test_predict_prod_mode_auth_success(client, monkeypatch, mock_app_dependencies):
    """Tests POST /predict in 'prod' mode with successful authentication."""
    monkeypatch.setattr("db.auth.ENV", "prod")
    mock_collection, mock_model, mock_verify_token = mock_app_dependencies

    response = client.post("/predict", params={"text": "hello prod"}, headers={"Authorization": "Bearer valid"})
    assert response.status_code == 200
    assert response.json()["owner"] == "mock_prod_user"
    
    mock_verify_token.assert_called_once()
    mock_model.predict.assert_called_once_with("hello prod")
    mock_collection.insert_one.assert_called_once()

def test_predict_prod_mode_auth_fail(client, monkeypatch, mock_app_dependencies):
    """Tests POST /predict in 'prod' mode with failed authentication."""
    monkeypatch.setattr("db.auth.ENV", "prod")
    mock_collection, mock_model, mock_verify_token = mock_app_dependencies
    mock_verify_token.side_effect = HTTPException(status_code=401, detail="Invalid Token")

    response = client.post("/predict", params={"text": "wont work"}, headers={"Authorization": "Bearer invalid"})
    assert response.status_code == 401
    assert "Invalid Token" in response.json()["detail"]
    
    mock_model.predict.assert_not_called()
    mock_collection.insert_one.assert_not_called()

def test_predict_no_models_loaded(client, monkeypatch, mock_app_dependencies):
    """Tests the edge case where no models are loaded."""
    monkeypatch.setattr("app.app.ENV", "dev")
    # Overwrite the mock for load_all_classifiers to return an empty dict for this test
    monkeypatch.setattr("app.services.load_all_classifiers", lambda urls: {})
    mock_collection, _, _ = mock_app_dependencies
    
    # Re-create client to trigger lifespan with the new mock
    with TestClient(app) as test_client:
        response = test_client.post("/predict", params={"text": "no models"})
    
    assert response.status_code == 200
    assert response.json()["predictions"] == {}
    mock_collection.insert_one.assert_called_once()


# --- Integration Test ---

@pytest.mark.integration
def test_integration_real_model_predict(monkeypatch, mock_app_dependencies):
    """
    Integration Test: Verifies the full prediction flow using a real model
    loaded from W&B via the app's lifespan event.
    """
    load_dotenv() 
    if not os.getenv("WANDB_API_KEY") or not os.getenv("WANDB_MODELS"):
        pytest.skip("WANDB_API_KEY or WANDB_MODELS not configured.")

    monkeypatch.setattr("app.app.ENV", "dev")
    
    # 1. Configure the app to load only the first model from the .env file
    first_model_url = os.getenv("WANDB_MODELS").split(',')[0].strip()
    model_name = first_model_url.split('/')[-1].split(':')[0]
    monkeypatch.setattr("app.app.get_model_urls", lambda: first_model_url)
    
    # 2. Create the TestClient, which triggers the lifespan event to load the real model
    with TestClient(app) as client:
        mock_collection, _, _ = mock_app_dependencies

        # 3. Make a prediction request
        test_text = "wait what?" # Assumes the first model is a 'confusion' classifier
        response = client.post("/predict", params={"text": test_text})
        
        # 4. Assertions
        assert response.status_code == 200
        data = response.json()
        
        assert model_name in data["predictions"]
        prediction = data["predictions"][model_name]["top_intent"]
        assert prediction == "confusion"
        
        mock_collection.insert_one.assert_called_once()
    
    print("\n[Integration Test] Passed: Real model loaded and predicted correctly.")
