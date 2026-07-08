"""
SupervisorAgent — analyzes query intent and decides routing.
Uses Claude via langchain-anthropic.
"""
from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.config import settings

SUPERVISOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a legal research supervisor. Your job is to route an incoming
query to the most appropriate agent pipeline.

Available routes:
- "full_pipeline"  → retrieval + classification + risk + summarization (default for general research questions)
- "retriever"      → semantic clause search only
- "classifier"     → classify a provided clause text only
- "risk"           → risk analysis of a provided clause text only
- "summarizer"     → summarize provided content only

Rules:
1. If the query asks for clause examples, patterns, or research → "full_pipeline"
2. If the query provides a specific clause and asks what type it is → "classifier"
3. If the query provides a specific clause and asks about risks → "risk"
4. If the query asks for a search or retrieval → "retriever"
5. If the query just asks to summarize already-known content → "summarizer"
6. When in doubt → "full_pipeline"

Respond with ONLY the route name, nothing else.
""",
        ),
        ("human", "Query: {original_query}"),
    ]
)


def create_supervisor_chain():
    llm = ChatAnthropic(
        model=settings.llm_model,
        temperature=0,
        anthropic_api_key=settings.anthropic_api_key,
        max_tokens=64,
    )
    return SUPERVISOR_PROMPT | llm | StrOutputParser()
