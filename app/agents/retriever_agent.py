"""
RetrieverAgent — semantic search over ChromaDB using direct tool calls.
Bypasses LangChain AgentExecutor to avoid Claude tool-calling prompt issues.
"""
from __future__ import annotations

import logging
from app.config import settings

logger = logging.getLogger(__name__)


def create_retriever_agent():
    """Returns a simple retriever that calls ChromaDB directly — no LLM needed."""

    class DirectRetriever:
        def invoke(self, inputs: dict) -> dict:
            query = inputs.get("input", "")
            try:
                from app.tools.vector_store import VectorStoreManager
                vs = VectorStoreManager()

                # Extract domain filter if embedded in query string
                domain_filter = None
                if "(domain:" in query:
                    parts = query.split("(domain:")
                    query = parts[0].strip()
                    domain_filter = parts[1].replace(")", "").strip()

                results = vs.similarity_search(
                    query,
                    n_results=5,
                    domain_filter=domain_filter,
                )

                if not results:
                    return {"output": "No relevant clauses found in the database."}

                # Format results clearly
                formatted = []
                for i, r in enumerate(results):
                    meta = r.get("metadata", {})
                    score = round(1 - r.get("distance", 1), 3)
                    formatted.append(
                        "[Result " + str(i + 1) + "] "
                        "Domain: " + meta.get("domain", "N/A") + " | "
                        "Type: " + meta.get("clause_type", "N/A") + " | "
                        "Similarity: " + str(score) + "\n"
                        + r.get("clause_text", "")[:600]
                    )

                return {"output": "\n\n---\n\n".join(formatted)}

            except Exception as e:
                logger.error("Retriever error: %s", e)
                return {"output": "[Retrieval failed: " + str(e) + "]"}

    return DirectRetriever()
