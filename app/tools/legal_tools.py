"""
LangChain custom tools that agents can call.
The VectorStoreManager is instantiated lazily so imports don't trigger
ChromaDB initialization at module load time.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from langchain.tools import tool

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_vs():
    from app.tools.vector_store import VectorStoreManager
    return VectorStoreManager()


@tool
def search_legal_clauses(query: str) -> str:
    """
    Search for the most relevant legal clauses across ALL domains
    using semantic (vector) similarity.  Returns up to 5 results with
    domain, clause type, and clause text.
    """
    vs = _get_vs()
    results = vs.similarity_search(query, n_results=5)
    if not results:
        return "No relevant clauses found. The vector store may be empty — run ingestion first."

    output = []
    for r in results:
        meta = r["metadata"]
        score = round(1 - r["distance"], 3)   # convert cosine distance → similarity
        output.append(
            f"[Domain: {meta.get('domain', 'N/A')} | "
            f"Type: {meta.get('clause_type', 'N/A')} | "
            f"Similarity: {score}]\n"
            f"{r['clause_text'][:600]}"
        )
    return "\n\n---\n\n".join(output)


@tool
def search_by_domain(query: str, domain: str) -> str:
    """
    Search legal clauses filtered to a SPECIFIC domain
    (e.g. 'bank-accounts', 'employment-contracts').
    Returns up to 5 results from that domain only.
    """
    vs = _get_vs()
    results = vs.similarity_search(query, n_results=5, domain_filter=domain)
    if not results:
        return f"No clauses found in domain '{domain}'. Check the domain name or run ingestion."

    output = []
    for r in results:
        meta = r["metadata"]
        score = round(1 - r["distance"], 3)
        output.append(
            f"[Type: {meta.get('clause_type', 'N/A')} | Similarity: {score}]\n"
            f"{r['clause_text'][:600]}"
        )
    return "\n\n---\n\n".join(output)
