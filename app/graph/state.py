from typing import TypedDict, Optional


class AgentState(TypedDict):
    """Shared state passed between all nodes in the LangGraph state machine."""

    # Input
    original_query: str

    # Agent outputs (populated progressively)
    retrieved_clauses: str
    classification: str
    risk_analysis: str
    final_summary: str

    # Routing
    next_agent: str

    # Optional scoping
    domain_filter: Optional[str]

    # Error capture
    error: Optional[str]
