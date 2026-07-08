"""
API integration tests — requires the FastAPI server to be running.
Run:  pytest tests/test_api.py -v
      (in a separate terminal: uvicorn app.main:app --reload --port 8000)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import pytest
import requests

BASE_URL = "http://localhost:8000"


def is_server_up() -> bool:
    try:
        return requests.get(f"{BASE_URL}/health", timeout=3).status_code == 200
    except Exception:
        return False


# Skip all tests if server isn't running
pytestmark = pytest.mark.skipif(
    not is_server_up(),
    reason="FastAPI server not running on localhost:8000",
)


# ── Health ─────────────────────────────────────────────────────────────────

def test_health():
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"


# ── Domains ────────────────────────────────────────────────────────────────

def test_list_domains():
    r = requests.get(f"{BASE_URL}/api/domains")
    assert r.status_code == 200
    data = r.json()
    assert "domains" in data
    assert "total" in data
    assert isinstance(data["domains"], list)


# ── Stats ──────────────────────────────────────────────────────────────────

def test_stats():
    r = requests.get(f"{BASE_URL}/api/stats")
    assert r.status_code == 200
    data = r.json()
    assert "total_vectors" in data
    assert "total_domains" in data


# ── Classify ───────────────────────────────────────────────────────────────

def test_classify_endpoint():
    payload = {
        "clause_text": (
            "The Customer shall indemnify and hold harmless the Company "
            "from any claims arising from Customer's breach of this Agreement."
        )
    }
    r = requests.post(f"{BASE_URL}/api/classify", json=payload, timeout=60)
    assert r.status_code == 200
    data = r.json()
    assert "classification" in data
    assert isinstance(data["classification"], str)
    assert len(data["classification"]) > 0


# ── Risk ───────────────────────────────────────────────────────────────────

def test_risk_endpoint():
    payload = {
        "clause_text": (
            "Company reserves the right to modify these terms at any time "
            "without prior notice to the customer."
        )
    }
    r = requests.post(f"{BASE_URL}/api/risk", json=payload, timeout=60)
    assert r.status_code == 200
    data = r.json()
    assert "risk_analysis" in data
    raw = data["risk_analysis"]
    cleaned = raw.strip().strip("```json").strip("```").strip()
    parsed = json.loads(cleaned)
    assert "risk_level" in parsed


# ── Full Query ─────────────────────────────────────────────────────────────

def test_full_query_endpoint():
    payload = {"question": "What are typical indemnification clauses?"}
    r = requests.post(f"{BASE_URL}/api/query", json=payload, timeout=180)
    assert r.status_code == 200
    data = r.json()
    required = {"original_query", "retrieved_clauses", "classification", "risk_analysis", "final_summary"}
    assert required.issubset(data.keys())


def test_full_query_with_domain_filter():
    payload = {
        "question": "indemnification obligations",
        "domain_filter": "bank-accounts",
    }
    r = requests.post(f"{BASE_URL}/api/query", json=payload, timeout=180)
    assert r.status_code == 200


def test_query_missing_question_returns_422():
    r = requests.post(f"{BASE_URL}/api/query", json={}, timeout=10)
    assert r.status_code == 422
