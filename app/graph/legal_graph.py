"""
Legal Research Pipeline — sequential execution, no LangGraph dependency.
Each step is called directly and errors are captured per-step.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def run_supervisor(query: str) -> str:
    """Decide routing. Returns: full_pipeline / retriever / classifier / risk / summarizer"""
    try:
        from app.agents.supervisor import create_supervisor_chain
        chain  = create_supervisor_chain()
        result = chain.invoke({"original_query": query})
        route  = result.strip().lower()
        valid  = {"full_pipeline", "retriever", "classifier", "risk", "summarizer"}
        route  = route if route in valid else "full_pipeline"
        logger.info("Supervisor → %s", route)
        return route
    except Exception as e:
        logger.error("Supervisor failed: %s", e)
        return "full_pipeline"


def run_retriever(query: str, domain_filter: str = None) -> str:
    """Direct ChromaDB similarity search. Returns formatted clause text."""
    try:
        from app.tools.vector_store import VectorStoreManager
        vs = VectorStoreManager()

        if vs.count() == 0:
            return "Vector store is empty. Please run data ingestion first."

        results = vs.similarity_search(query, n_results=5, domain_filter=domain_filter)

        if not results:
            return "No relevant clauses found for this query."

        formatted = []
        for i, r in enumerate(results):
            meta  = r.get("metadata", {})
            score = round(1 - r.get("distance", 1), 3)
            formatted.append(
                "[Result " + str(i + 1) + "] "
                "Domain: " + meta.get("domain", "N/A") + " | "
                "Type: "   + meta.get("clause_type", "N/A") + " | "
                "Similarity: " + str(score) + "\n"
                + r.get("clause_text", "")[:600]
            )

        logger.info("Retriever found %d results", len(results))
        return "\n\n---\n\n".join(formatted)

    except Exception as e:
        logger.error("Retriever failed: %s", e)
        return "[Retrieval failed: " + str(e) + "]"


def run_classifier(text: str) -> str:
    """Classify clause type. Returns category label string."""
    try:
        from app.agents.classifier_agent import create_classifier_chain
        chain  = create_classifier_chain()
        result = chain.invoke({"clause_text": text[:3000]})
        logger.info("Classifier → %s", result)
        return result
    except Exception as e:
        logger.error("Classifier failed: %s", e)
        return "other"


def run_risk(text: str) -> str:
    """Risk analysis. Returns JSON string."""
    try:
        from app.agents.risk_agent import create_risk_chain
        chain  = create_risk_chain()
        result = chain.invoke({"clause_text": text[:3000]})
        logger.info("Risk agent completed")
        return result
    except Exception as e:
        logger.error("Risk agent failed: %s", e)
        return '{"risk_level":"unknown","risk_factors":[],"recommendation":"Analysis failed: ' + str(e) + '"}'


def run_summarizer(original_query: str, retrieved_clauses: str,
                   classification: str, risk_analysis: str) -> str:
    """Generate plain-English summary. Returns markdown string."""
    try:
        from app.agents.summarizer_agent import create_summarizer_chain
        chain  = create_summarizer_chain()
        result = chain.invoke({
            "original_query":    original_query,
            "retrieved_clauses": retrieved_clauses,
            "classification":    classification,
            "risk_analysis":     risk_analysis,
        })
        logger.info("Summarizer completed")
        return result
    except Exception as e:
        logger.error("Summarizer failed: %s", e)
        return "[Summary failed: " + str(e) + "]"


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(initial_state: dict) -> dict:
    """
    Runs the full pipeline sequentially.
    Each step result is stored directly in the state dict.
    Errors in any step are captured and stored — pipeline always completes.
    """
    state = {
        "original_query":    initial_state.get("original_query", ""),
        "retrieved_clauses": "",
        "classification":    "",
        "risk_analysis":     "",
        "final_summary":     "",
        "next_agent":        "",
        "domain_filter":     initial_state.get("domain_filter"),
        "error":             None,
    }

    query         = state["original_query"]
    domain_filter = state["domain_filter"]

    # ── Step 1: Supervisor ────────────────────────────────────────────────
    logger.info("=== Pipeline start: %s ===", query[:80])
    route = run_supervisor(query)
    state["next_agent"] = route

    # ── Step 2: Retriever (always runs) ───────────────────────────────────
    logger.info("Step 2: Retriever")
    state["retrieved_clauses"] = run_retriever(query, domain_filter)

    if route == "retriever":
        logger.info("Route=retriever, stopping after retrieval")
        return state

    # ── Step 3: Classifier ────────────────────────────────────────────────
    logger.info("Step 3: Classifier")
    text_to_analyse = state["retrieved_clauses"] if state["retrieved_clauses"] else query
    state["classification"] = run_classifier(text_to_analyse)

    if route == "classifier":
        logger.info("Route=classifier, stopping after classification")
        return state

    # ── Step 4: Risk ──────────────────────────────────────────────────────
    logger.info("Step 4: Risk agent")
    state["risk_analysis"] = run_risk(text_to_analyse)

    if route == "risk":
        logger.info("Route=risk, stopping after risk analysis")
        return state

    # ── Step 5: Summarizer ────────────────────────────────────────────────
    logger.info("Step 5: Summarizer")
    state["final_summary"] = run_summarizer(
        original_query    = query,
        retrieved_clauses = state["retrieved_clauses"],
        classification    = state["classification"],
        risk_analysis     = state["risk_analysis"],
    )

    logger.info("=== Pipeline complete ===")
    return state


# ── Compatibility shim ────────────────────────────────────────────────────────

class _PipelineCompat:
    """Drop-in replacement for compiled LangGraph — keeps routes.py unchanged."""
    def invoke(self, state: dict) -> dict:
        return run_pipeline(state)


def build_legal_graph() -> _PipelineCompat:
    return _PipelineCompat()
