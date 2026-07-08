"""
One-time data ingestion script.
Run from the project root:  python scripts/ingest.py
"""
import sys
import logging
from pathlib import Path

# Make sure the project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    from app.tools.csv_loader import load_all_clauses
    from app.tools.vector_store import VectorStoreManager

    logger.info("=" * 60)
    logger.info("  Multi-Agent Legal Research — Data Ingestion Script")
    logger.info("=" * 60)

    logger.info("Step 1: Loading CSV files...")
    docs = load_all_clauses()

    if not docs:
        logger.error(
            "No documents loaded. "
            "Please place your CSV files in data/legal_clauses/ and try again."
        )
        sys.exit(1)

    logger.info(f"Step 2: Ingesting {len(docs)} clauses into ChromaDB...")
    vs = VectorStoreManager()
    total = vs.ingest_documents(docs)

    logger.info("=" * 60)
    logger.info(f"  Ingestion complete! {total} clauses stored in ChromaDB.")
    logger.info(f"  Vector store location: {vs.chroma_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
