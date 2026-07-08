from pydantic import BaseModel
from typing import Optional, List


class QueryRequest(BaseModel):
    question: str
    domain_filter: Optional[str] = None


class ClassifyRequest(BaseModel):
    clause_text: str


class RiskRequest(BaseModel):
    clause_text: str


class SummarizeRequest(BaseModel):
    clause_text: str


class QueryResponse(BaseModel):
    original_query: str
    retrieved_clauses: str = ""
    classification: str = ""
    risk_analysis: str = ""
    final_summary: str = ""


class ClassifyResponse(BaseModel):
    clause_text: str
    classification: str


class RiskResponse(BaseModel):
    clause_text: str
    risk_analysis: str


class DomainListResponse(BaseModel):
    domains: List[str]
    total: int


class IngestResponse(BaseModel):
    status: str
    documents_ingested: int
    message: str = ""
