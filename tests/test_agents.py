"""
Unit tests for individual agent chains (Claude / Anthropic).
Run:  pytest tests/test_agents.py -v

NOTE: These tests make real Anthropic API calls.
      Set ANTHROPIC_API_KEY in your .env before running.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import pytest


# ── Classifier ─────────────────────────────────────────────────────────────

def test_classifier_known_clause():
    from app.agents.classifier_agent import create_classifier_chain
    chain = create_classifier_chain()
    result = chain.invoke({
        "clause_text": (
            "The Customer shall indemnify, defend, and hold harmless the Company "
            "from any claims, damages, or liabilities arising from Customer's use."
        )
    })
    assert isinstance(result, str)
    assert result.strip().lower() in [
        "indemnification", "liability", "other"
    ], f"Unexpected classification: {result}"


def test_classifier_returns_single_word():
    from app.agents.classifier_agent import create_classifier_chain
    chain = create_classifier_chain()
    result = chain.invoke({
        "clause_text": "Either party may terminate this Agreement upon 30 days notice."
    })
    # Claude should return a single word or underscored label
    words = result.strip().split()
    assert len(words) == 1 or "_" in result.strip(), \
        f"Expected single-word/underscore label, got: {result!r}"


# ── Risk Agent ─────────────────────────────────────────────────────────────

def test_risk_agent_returns_json():
    from app.agents.risk_agent import create_risk_chain
    chain = create_risk_chain()
    result = chain.invoke({
        "clause_text": (
            "Company may modify these terms at any time without notice. "
            "Customer's continued use constitutes acceptance."
        )
    })
    cleaned = result.strip().strip("```json").strip("```").strip()
    parsed = json.loads(cleaned)
    assert "risk_level" in parsed
    assert parsed["risk_level"] in ["low", "medium", "high"]
    assert "risk_factors" in parsed
    assert isinstance(parsed["risk_factors"], list)


def test_risk_agent_high_risk_clause():
    from app.agents.risk_agent import create_risk_chain
    chain = create_risk_chain()
    result = chain.invoke({
        "clause_text": (
            "Customer assumes unlimited liability for all losses, damages, and costs "
            "of any kind whatsoever, without any cap or limitation."
        )
    })
    cleaned = result.strip().strip("```json").strip("```").strip()
    parsed = json.loads(cleaned)
    assert parsed["risk_level"] in ["medium", "high"]


# ── Summarizer ─────────────────────────────────────────────────────────────

def test_summarizer_returns_text():
    from app.agents.summarizer_agent import create_summarizer_chain
    chain = create_summarizer_chain()
    result = chain.invoke({
        "original_query": "What is an indemnification clause?",
        "retrieved_clauses": "Customer shall indemnify company from all claims.",
        "classification": "indemnification",
        "risk_analysis": '{"risk_level": "medium", "risk_factors": ["broad scope"], "recommendation": "Negotiate cap"}',
    })
    assert isinstance(result, str)
    assert len(result) > 50, "Summary is too short"


# ── Supervisor ─────────────────────────────────────────────────────────────

def test_supervisor_routes_general_query():
    from app.agents.supervisor import create_supervisor_chain
    chain = create_supervisor_chain()
    result = chain.invoke({
        "original_query": "What are typical termination clauses in employment contracts?"
    })
    route = result.strip().lower()
    valid = {"full_pipeline", "retriever", "classifier", "risk", "summarizer"}
    assert route in valid, f"Invalid route returned: {route!r}"


def test_supervisor_routes_classify_query():
    from app.agents.supervisor import create_supervisor_chain
    chain = create_supervisor_chain()
    result = chain.invoke({
        "original_query": "Classify this clause: 'Customer shall indemnify company from all claims.'"
    })
    route = result.strip().lower()
    assert route in {"classifier", "full_pipeline"}
