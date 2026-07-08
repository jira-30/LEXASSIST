"""
Unit tests for the VectorStoreManager.
Run:  pytest tests/test_vector_store.py -v

NOTE: These tests require ChromaDB to be installed and writable tmp storage.
      They do NOT require the real 395-CSV dataset — they use synthetic data.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import tempfile

from app.tools.vector_store import VectorStoreManager

SAMPLE_DOCS = [
    {
        "id": "test_0",
        "clause_text": "The customer shall indemnify and hold harmless the company from any and all claims.",
        "clause_type": "indemnification",
        "domain": "test-domain",
        "source_file": "test.csv",
    },
    {
        "id": "test_1",
        "clause_text": "Either party may terminate this agreement upon thirty days written notice.",
        "clause_type": "termination",
        "domain": "test-domain",
        "source_file": "test.csv",
    },
    {
        "id": "test_2",
        "clause_text": "All confidential information shall be kept secret and not disclosed to third parties.",
        "clause_type": "confidentiality",
        "domain": "test-domain",
        "source_file": "test.csv",
    },
]


@pytest.fixture
def vs(tmp_path):
    """Create a fresh VectorStoreManager backed by a temp directory."""
    return VectorStoreManager(chroma_path=str(tmp_path / "test_chroma"))


def test_ingest_documents(vs):
    total = vs.ingest_documents(SAMPLE_DOCS)
    assert total == len(SAMPLE_DOCS)
    assert vs.count() == len(SAMPLE_DOCS)


def test_similarity_search_returns_results(vs):
    vs.ingest_documents(SAMPLE_DOCS)
    results = vs.similarity_search("indemnification hold harmless", n_results=2)
    assert len(results) > 0
    assert "clause_text" in results[0]
    assert "metadata" in results[0]
    assert "distance" in results[0]


def test_similarity_search_domain_filter(vs):
    vs.ingest_documents(SAMPLE_DOCS)
    results = vs.similarity_search("terminate agreement", n_results=3, domain_filter="test-domain")
    assert len(results) > 0
    for r in results:
        assert r["metadata"]["domain"] == "test-domain"


def test_similarity_search_empty_store(vs):
    """Should return empty list, not raise, when store is empty."""
    results = vs.similarity_search("anything")
    assert results == []


def test_count_after_ingest(vs):
    assert vs.count() == 0
    vs.ingest_documents(SAMPLE_DOCS)
    assert vs.count() == len(SAMPLE_DOCS)


def test_upsert_deduplication(vs):
    """Upserting the same docs twice should not double the count."""
    vs.ingest_documents(SAMPLE_DOCS)
    vs.ingest_documents(SAMPLE_DOCS)
    assert vs.count() == len(SAMPLE_DOCS)


def test_delete_collection(vs):
    vs.ingest_documents(SAMPLE_DOCS)
    assert vs.count() > 0
    vs.delete_collection()
    assert vs.count() == 0
