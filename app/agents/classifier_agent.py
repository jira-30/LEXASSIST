"""
ClassifierAgent — classifies a legal clause against ALL 395 dataset categories.

Strategy:
1. PRIMARY:  Semantic search in ChromaDB — find the most similar stored clause
             and return its clause_type. This uses the ACTUAL labels from your
             395 CSV files, so it covers every category automatically.
2. FALLBACK: If ChromaDB search fails or returns low confidence, ask the LLM
             with the full dynamic category list derived from CSV filenames.
"""
from __future__ import annotations

import os
import logging
from pathlib import Path
from functools import lru_cache

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE         = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
_DATA_DIR     = _PROJECT_ROOT / "data" / "legal_clauses"


# ── Dynamically load ALL category names from CSV filenames ────────────────────
@lru_cache(maxsize=1)
def get_all_categories() -> list:
    """
    Reads the data/legal_clauses directory and returns every CSV stem
    as a category name (hyphens → underscores, lowercased).
    Result is cached after first call.
    """
    if not _DATA_DIR.exists():
        return ["other"]
    categories = sorted(
        f.stem.replace("-", "_").lower()
        for f in _DATA_DIR.glob("*.csv")
    )
    return categories if categories else ["other"]


# ── Primary classifier: ChromaDB semantic similarity ─────────────────────────

def classify_via_chromadb(clause_text: str, threshold: float = 0.55) -> str | None:
    """
    Searches ChromaDB for the most similar stored clause.
    Returns the clause_type metadata of the top result if similarity >= threshold.
    Returns None if confidence is too low or search fails.
    """
    try:
        from app.tools.vector_store import VectorStoreManager
        vs      = VectorStoreManager()
        results = vs.similarity_search(clause_text, n_results=1)

        if not results:
            return None

        top       = results[0]
        distance  = top.get("distance", 1.0)
        similarity = 1 - distance

        logger.info("ChromaDB top match similarity: %.3f", similarity)

        if similarity >= threshold:
            raw_type = top.get("metadata", {}).get("clause_type", "")
            if raw_type and raw_type.lower() != "unknown":
                # Normalise: hyphens → underscores, lowercase
                return raw_type.strip().lower().replace("-", "_").replace(" ", "_")

        return None

    except Exception as e:
        logger.error("ChromaDB classification failed: %s", e)
        return None


# ── Fallback classifier: LLM with full dynamic category list ──────────────────

def classify_via_llm(clause_text: str) -> str:
    """
    Asks the LLM to classify the clause using the full list of categories
    derived from all CSV filenames. Falls back to 'other' on any error.
    """
    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from app.config import settings

        categories     = get_all_categories()
        categories_str = "\n".join("- " + c for c in categories)

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a precise legal clause classifier.\n\n"
                "Classify the clause into EXACTLY ONE category from this list:\n\n"
                + categories_str + "\n\n"
                "Rules:\n"
                "- Output ONLY the category name, nothing else\n"
                "- Use underscores not hyphens\n"
                "- Must exactly match one category from the list above\n"
                "- If truly none match, output: other\n"
            ),
            ("human", "Clause:\n{clause_text}"),
        ])

        llm   = ChatAnthropic(
            model=settings.llm_model,
            temperature=0,
            anthropic_api_key=settings.anthropic_api_key,
            max_tokens=32,
        )
        chain  = prompt | llm | StrOutputParser()
        result = chain.invoke({"clause_text": clause_text[:3000]})
        label  = result.strip().lower().replace("-", "_").replace(" ", "_")

        # Validate against known categories
        all_cats = get_all_categories()
        if label in all_cats:
            return label

        # Fuzzy match — check if result is contained in any category
        for cat in all_cats:
            if label in cat or cat in label:
                return cat

        return "other"

    except Exception as e:
        logger.error("LLM classification failed: %s", e)
        return "other"


# ── Public interface ──────────────────────────────────────────────────────────

def create_classifier_chain():
    """
    Returns a chain-compatible object with .invoke({"clause_text": ...}) -> str.
    Uses ChromaDB similarity as primary, LLM as fallback.
    """

    class SmartClassifier:
        def invoke(self, inputs: dict) -> str:
            clause_text = inputs.get("clause_text", "").strip()
            if not clause_text:
                return "other"

            # Step 1 — Try ChromaDB semantic match (fast, free, uses real labels)
            result = classify_via_chromadb(clause_text)
            if result:
                logger.info("Classified via ChromaDB: %s", result)
                return result

            # Step 2 — Fall back to LLM with full category list
            logger.info("ChromaDB confidence low, falling back to LLM")
            result = classify_via_llm(clause_text)
            logger.info("Classified via LLM: %s", result)
            return result

    return SmartClassifier()
