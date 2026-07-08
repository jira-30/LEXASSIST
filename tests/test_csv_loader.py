"""
Unit tests for the CSV loader.
Run:  pytest tests/test_csv_loader.py -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest
import tempfile
import os

from app.tools.csv_loader import detect_column, load_all_clauses


# ── detect_column ──────────────────────────────────────────────────────────

def test_detect_column_exact_match():
    df = pd.DataFrame({"clause_text": ["hello"], "type": ["liability"]})
    assert detect_column(df, ["clause_text", "text"]) == "clause_text"


def test_detect_column_case_insensitive():
    df = pd.DataFrame({"Clause_Text": ["hello"], "Type": ["liability"]})
    assert detect_column(df, ["clause_text"]) == "Clause_Text"


def test_detect_column_fallback_to_object():
    df = pd.DataFrame({"col_a": [1, 2], "col_b": ["text1", "text2"]})
    result = detect_column(df, ["nonexistent"])
    assert result == "col_b"


def test_detect_column_no_match_returns_none():
    df = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
    assert detect_column(df, ["nonexistent"]) is None


# ── load_all_clauses ────────────────────────────────────────────────────────

def test_load_all_clauses_returns_list():
    """Should return a list (empty is fine if data dir missing)."""
    docs = load_all_clauses()
    assert isinstance(docs, list)


def test_load_all_clauses_doc_structure():
    """Each document must have required keys."""
    docs = load_all_clauses()
    if docs:
        required_keys = {"id", "clause_text", "clause_type", "domain", "source_file"}
        for doc in docs[:5]:
            assert required_keys.issubset(doc.keys()), f"Missing keys in: {doc.keys()}"


def test_load_all_clauses_no_short_texts():
    """No document should have clause_text shorter than 20 chars."""
    docs = load_all_clauses()
    for doc in docs:
        assert len(doc["clause_text"]) >= 20, f"Short text found: {doc['clause_text']!r}"


def test_load_all_clauses_unique_ids():
    """All document IDs should be unique."""
    docs = load_all_clauses()
    ids = [d["id"] for d in docs]
    assert len(ids) == len(set(ids)), "Duplicate IDs found in loaded documents"
