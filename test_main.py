from fastapi.testclient import TestClient
from main import app


# Test 1: Check if GET /health works and returns 200 OK
def test_health_check():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


# Test 2: Check a valid billing complaint returns 200 and expected fields
def test_predict_valid_billing():
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": "I was charged twice"})
        assert response.status_code == 200
        data = response.json()
        assert data["category"] == "Billing"
        assert data["priority"] == "High"
        assert data["confidence"] == 0.93


# Test 3: Check a valid general message returns 200
def test_predict_valid_general():
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": "How do I reset my password?"})
        assert response.status_code == 200
        data = response.json()
        assert data["category"] == "General"
        assert data["priority"] == "Low"


# Test 4: Check that empty text "" returns 422 Unprocessable Entity
def test_predict_empty_text_returns_422():
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": ""})
        assert response.status_code == 422


# Test 5: Check that spaces-only text "   " returns 422 Unprocessable Entity
def test_predict_whitespace_returns_422():
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": "   "})
        assert response.status_code == 422
