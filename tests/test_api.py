"""FastAPI endpoint smoke tests."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_health_returns_expected_keys():
  res = client.get("/health")
  assert res.status_code == 200
  body = res.json()
  assert "status" in body
  assert "keys_loaded" in body
  assert "keys_status" in body
  assert "index_loaded" in body
  assert "drug_count" in body


def test_chat_rejects_empty_message():
  res = client.post("/chat", json={"message": "   ", "history": []})
  assert res.status_code == 400


@patch("app.rag")
def test_chat_returns_response(mock_rag):
  mock_rag.return_value = "أهلاً بك"
  res = client.post("/chat", json={"message": "عندي برد", "history": []})
  assert res.status_code == 200
  assert res.json()["response"] == "أهلاً بك"
