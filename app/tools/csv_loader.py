import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Resolve DATA_DIR relative to this file's location (works from any cwd)
_HERE = Path(__file__).resolve().parent          # app/tools/
_PROJECT_ROOT = _HERE.parent.parent              # project root
DATA_DIR = _PROJECT_ROOT / "data" / "legal_clauses"

# Common column-name variants found across the 395 CSVs
CLAUSE_TEXT_VARIANTS = [
    "clause_text", "text", "clause", "content",
    "description", "sentence", "passage", "body",
]
CLAUSE_TYPE_VARIANTS = [
    "clause_type", "type", "label", "category",
    "class", "tag", "annotation",
]


def detect_column(df: pd.DataFrame, variants: List[str]) -> Optional[str]:
    """Return the first matching column name from a list of variants (case-insensitive)."""
    lower_cols = {c.lower(): c for c in df.columns}
    for v in variants:
        if v.lower() in lower_cols:
            return lower_cols[v.lower()]
    # Fallback: first object-dtype column
    for col in df.columns:
        if df[col].dtype == object:
            return col
    return None


def load_all_clauses() -> List[Dict]:
    """
    Load all CSV files from DATA_DIR and return a flat list of clause documents.
    Each document dict contains: id, clause_text, clause_type, domain, source_file.
    """
    if not DATA_DIR.exists():
        logger.error(f"Data directory not found: {DATA_DIR}")
        return []

    all_docs: List[Dict] = []
    csv_files = list(DATA_DIR.glob("*.csv"))

    if not csv_files:
        logger.warning(f"No CSV files found in {DATA_DIR}")
        return []

    logger.info(f"Found {len(csv_files)} CSV files in {DATA_DIR}")

    for csv_file in csv_files:
        domain = csv_file.stem  # e.g. "bank-accounts"
        try:
            df = pd.read_csv(
                csv_file,
                encoding="utf-8",
                on_bad_lines="skip",
                low_memory=False,
            )
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(
                    csv_file,
                    encoding="latin-1",
                    on_bad_lines="skip",
                    low_memory=False,
                )
            except Exception as e:
                logger.warning(f"Could not load {csv_file.name}: {e}")
                continue
        except Exception as e:
            logger.warning(f"Could not load {csv_file.name}: {e}")
            continue

        text_col = detect_column(df, CLAUSE_TEXT_VARIANTS)
        type_col = detect_column(df, CLAUSE_TYPE_VARIANTS)

        if not text_col:
            logger.warning(f"No text column found in {csv_file.name} — skipping")
            continue

        for idx, row in df.iterrows():
            text = str(row[text_col]).strip()
            if len(text) < 20:
                continue

            clause_type = "unknown"
            if type_col and pd.notna(row.get(type_col)):
                clause_type = str(row[type_col]).strip()

            doc = {
                "id": f"{domain}_{idx}",
                "clause_text": text,
                "clause_type": clause_type,
                "domain": domain,
                "source_file": csv_file.name,
            }
            all_docs.append(doc)

    logger.info(f"Loaded {len(all_docs)} clauses from {len(csv_files)} files")
    return all_docs


def get_available_domains() -> List[str]:
    """Return a sorted list of domain names derived from CSV filenames."""
    if not DATA_DIR.exists():
        return []
    return sorted(
        f.stem for f in DATA_DIR.glob("*.csv")
    )
