import pytest
from fastapi.testclient import TestClient
from main import app

def test_health_check():
    """Test the /health endpoint"""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

def test_chat_endpoint_valid_query():
    """Test the /api/chat endpoint with a valid query requiring tools"""
    with TestClient(app) as client:
        payload = {"query": "What is the capital of Peru?"}
        response = client.post("/api/chat", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        print(f"\n[API TEST OUTPUT] {data}\n")
        assert "answer" in data
        assert data["is_relevant"] is True
        assert "tools_used" in data
        assert isinstance(data["tools_used"], list)

def test_chat_endpoint_irrelevant_query():
    """Test the /api/chat endpoint with an irrelevant query to trigger the guardrail"""
    with TestClient(app) as client:
        payload = {"query": "How do I bake a chocolate cake?"}
        response = client.post("/api/chat", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        print(f"\n[API TEST OUTPUT] {data}\n")
        assert "answer" in data
        assert data["is_relevant"] is False
        assert data["tools_used"] == []
