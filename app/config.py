import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Anthropic
    anthropic_api_key: str = ""

    # ChromaDB & data paths
    chroma_db_path: str = "./chroma_db"
    data_dir: str = "./data/legal_clauses"

    # Models
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "claude-haiku-4-5-20251001"

    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# No lru_cache — always reads fresh from .env
settings = Settings()
