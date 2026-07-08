"""
FastAPI application entry point.
Run with: uvicorn app.main:app --reload --port 8000
"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

app = FastAPI(
    title="Multi-Agent Legal Research System",
    description=(
        "LangChain + LangGraph powered legal clause analysis API. "
        "Supports semantic search, clause classification, risk analysis, "
        "and plain-English summarization across 395 legal domains."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "ok",
        "message": "Multi-Agent Legal Research System is running",
        "version": "1.0.0",
    }
