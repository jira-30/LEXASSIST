"""
FastAPI route handlers.
"""
from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException
from app.models.schemas import (
    QueryRequest, QueryResponse,
    ClassifyRequest, ClassifyResponse,
    RiskRequest, RiskResponse,
    DomainListResponse, IngestResponse,
)
from app.tools.csv_loader import get_available_domains, load_all_clauses

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Legal Research"])

# ── Lazy pipeline loader ──────────────────────────────────────────────────────
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from app.graph.legal_graph import build_legal_graph
        _pipeline = build_legal_graph()
    return _pipeline


def _blank_state(question: str, domain_filter) -> dict:
    return {
        "original_query":    question,
        "retrieved_clauses": "",
        "classification":    "",
        "risk_analysis":     "",
        "final_summary":     "",
        "next_agent":        "",
        "domain_filter":     domain_filter,
        "error":             None,
    }


# ── Debug / health endpoints ──────────────────────────────────────────────────

@router.get("/debug")
async def debug_info():
    from app.config import settings
    from app.tools.vector_store import VectorStoreManager
    key = settings.anthropic_api_key
    key_preview = (key[:8] + "..." + key[-6:]) if len(key) > 14 else "NOT SET"
    try:
        vs = VectorStoreManager()
        vector_count = vs.count()
    except Exception as e:
        vector_count = "ERROR: " + str(e)
    csv_domains = get_available_domains()
    return {
        "api_key_preview": key_preview,
        "api_key_length":  len(key),
        "api_key_ok":      key.startswith("sk-ant-") and '"' not in key,
        "model":           settings.llm_model,
        "vector_count":    vector_count,
        "csv_files_found": len(csv_domains),
        "needs_ingestion": vector_count == 0 if isinstance(vector_count, int) else True,
    }


@router.get("/test-llm")
async def test_llm():
    """Calls the LLM with a minimal prompt to verify the API key works."""
    try:
        from app.config import settings
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(
            model=settings.llm_model,
            anthropic_api_key=settings.anthropic_api_key,
            max_tokens=32,
            temperature=0,
        )
        response = llm.invoke("Reply with exactly: LLM_OK")
        return {"status": "ok", "response": response.content}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@router.get("/test-query")
async def test_query():
    """
    Runs a minimal hardcoded query through the full pipeline.
    Use this to verify end-to-end before trying the UI.
    """
    try:
        from app.graph.legal_graph import run_pipeline
        state = _blank_state("What are typical indemnification clauses?", None)
        result = run_pipeline(state)
        return {
            "next_agent":        result.get("next_agent", ""),
            "retrieved_clauses": result.get("retrieved_clauses", "")[:300],
            "classification":    result.get("classification", ""),
            "risk_level":        "see risk_analysis field",
            "final_summary":     result.get("final_summary", "")[:500],
            "error":             result.get("error"),
        }
    except Exception as e:
        logger.exception("test-query failed")
        return {"status": "error", "detail": str(e)}


# ── Main endpoints ────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def full_research_query(request: QueryRequest):
    try:
        state  = _blank_state(request.question, request.domain_filter)
        result = get_pipeline().invoke(state)
        return QueryResponse(
            original_query    = result.get("original_query",    request.question),
            retrieved_clauses = result.get("retrieved_clauses", ""),
            classification    = result.get("classification",    ""),
            risk_analysis     = result.get("risk_analysis",     ""),
            final_summary     = result.get("final_summary",     ""),
        )
    except Exception as e:
        logger.exception("Error in full_research_query")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/classify", response_model=ClassifyResponse)
async def classify_clause(request: ClassifyRequest):
    try:
        from app.agents.classifier_agent import create_classifier_chain
        chain = create_classifier_chain()
        label = chain.invoke({"clause_text": request.clause_text})
        return ClassifyResponse(
            clause_text    = request.clause_text,
            classification = label.strip(),
        )
    except Exception as e:
        logger.exception("Error in classify_clause")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/risk", response_model=RiskResponse)
async def analyze_risk(request: RiskRequest):
    try:
        from app.agents.risk_agent import create_risk_chain
        chain    = create_risk_chain()
        analysis = chain.invoke({"clause_text": request.clause_text})
        return RiskResponse(
            clause_text   = request.clause_text,
            risk_analysis = analysis,
        )
    except Exception as e:
        logger.exception("Error in analyze_risk")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/domains", response_model=DomainListResponse)
async def list_domains():
    domains = get_available_domains()
    return DomainListResponse(domains=domains, total=len(domains))


@router.get("/stats")
async def get_stats():
    try:
        from app.tools.vector_store import VectorStoreManager
        vs      = VectorStoreManager()
        domains = get_available_domains()
        return {
            "total_vectors":  vs.count(),
            "total_domains":  len(domains),
            "domains_sample": domains[:10],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest", response_model=IngestResponse)
async def ingest_data():
    try:
        from app.tools.vector_store import VectorStoreManager
        docs = load_all_clauses()
        if not docs:
            return IngestResponse(
                status="warning",
                documents_ingested=0,
                message="No documents found. Place CSV files in data/legal_clauses/.",
            )
        vs    = VectorStoreManager()
        count = vs.ingest_documents(docs)
        return IngestResponse(
            status="success",
            documents_ingested=count,
            message="Successfully ingested {} clauses into ChromaDB.".format(count),
        )
    except Exception as e:
        logger.exception("Error during ingestion")
        raise HTTPException(status_code=500, detail=str(e))
