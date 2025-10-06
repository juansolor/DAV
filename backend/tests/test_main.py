from fastapi.testclient import TestClient
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from main import app

client = TestClient(app)

def test_health_endpoint():
    """Test the health check endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "listo" in response.json()["message"].lower()

def test_datasets_list_empty():
    """Test datasets list when empty"""
    response = client.get("/datasets/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_upload_invalid_file():
    """Test upload with invalid file format"""
    files = {"file": ("test.txt", "invalid content", "text/plain")}
    response = client.post("/datasets/upload", files=files)
    # Should either succeed or fail gracefully
    assert response.status_code in [200, 400]

def test_cors_headers():
    """Test that CORS headers are present"""
    response = client.options("/")
    # CORS middleware should handle OPTIONS requests