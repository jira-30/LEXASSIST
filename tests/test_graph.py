"""
Integration tests for the full LangGraph state machine.
Run:  pytest tests/test_graph.py -v

NOTE: These tests make real OpenAI API calls and require ChromaDB to be
      populated.  Run scripts/ingest.py first, or the retriever will return
      empty results (which is still a valid — but less useful — test).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from app.graph.legal_graph import build_legal_graph
from app.graph.state import AgentState


def make_state(query: str, domain: str = None) -> AgentState:
    return {
        "original_query": query,
        "retrieved_clauses": "",
        "classification": "",
        "risk_analysis": "",
        "final_summary": "",
        "next_agent": "",
        "domain_filter": domain,
        "error": None,
    }


@pytest.fixture(scope="module")
def graph():
    return build_legal_graph()


def test_graph_compiles(graph):
    assert graph is not None


def test_full_pipeline_returns_state(graph):
    state = make_state("What are termination clauses in employment contracts?")
    result = graph.invoke(state)
    assert isinstance(result, dict)
    assert "original_query" in result
    assert "next_agent" in result


def test_full_pipeline_has_summary(graph):
    state = make_state("Explain indemnification clauses in bank agreements.")
    result = graph.invoke(state)
    # Summary might be empty if no data ingested, but key must exist
    assert "final_summary" in result


def test_full_pipeline_has_classification(graph):
    state = make_state("What are warranty clauses?")
    result = graph.invoke(state)
    assert "classification" in result


def test_full_pipeline_has_risk_analysis(graph):
    state = make_state("Find risky liability clauses.")
    result = graph.invoke(state)
    assert "risk_analysis" in result


def test_graph_handles_short_query(graph):
    """Graph should not crash on a very short query."""
    state = make_state("liability")
    result = graph.invoke(state)
    assert isinstance(result, dict)


def test_graph_with_domain_filter(graph):
    state = make_state("indemnification obligations", domain="bank-accounts")
    result = graph.invoke(state)
    assert isinstance(result, dict)
    assert result.get("error") is None or result.get("error") == ""
