import logging
from pathlib import Path
from typing import List, Dict, Optional

import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
CHROMA_PATH = str(_PROJECT_ROOT / "chroma_db")
COLLECTION_NAME = "legal_clauses"


class VectorStoreManager:
    """Manages ChromaDB persistence, ingestion, and similarity search."""

    def __init__(self, chroma_path: str = CHROMA_PATH):
        self.chroma_path = chroma_path
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.embedding_fn = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB collection '{COLLECTION_NAME}' ready — "
            f"{self.collection.count()} documents stored."
        )

    # ── Ingestion ──────────────────────────────────────────────────────────

    def ingest_documents(self, docs: List[Dict], batch_size: int = 500) -> int:
        """Upsert documents into ChromaDB in batches. Returns total ingested."""
        total = 0
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            self.collection.upsert(
                ids=[d["id"] for d in batch],
                documents=[d["clause_text"] for d in batch],
                metadatas=[
                    {
                        "clause_type": d["clause_type"],
                        "domain": d["domain"],
                        "source_file": d["source_file"],
                    }
                    for d in batch
                ],
            )
            total += len(batch)
            logger.info(
                f"Ingested batch {i // batch_size + 1} "
                f"({total}/{len(docs)} docs)"
            )
        return total

    # ── Search ─────────────────────────────────────────────────────────────

    def similarity_search(
        self,
        query: str,
        n_results: int = 5,
        domain_filter: Optional[str] = None,
    ) -> List[Dict]:
        """
        Cosine-similarity search with optional domain filter.
        Returns list of {clause_text, metadata, distance}.
        """
        if self.collection.count() == 0:
            logger.warning("ChromaDB collection is empty — run ingestion first.")
            return []

        # chromadb 0.4.x uses plain dict for where (no $eq wrapper needed)
        where = {"domain": domain_filter} if domain_filter else None

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, self.collection.count()),
                where=where,
            )
        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}")
            return []

        docs = []
        for i, doc_text in enumerate(results["documents"][0]):
            docs.append(
                {
                    "clause_text": doc_text,
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                }
            )
        return docs

    def count(self) -> int:
        return self.collection.count()

    def delete_collection(self):
        """Full reset — delete and recreate the collection."""
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Collection reset.")
