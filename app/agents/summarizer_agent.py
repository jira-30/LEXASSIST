"""
SummarizerAgent — converts legal findings into plain-English summaries.
"""
from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.config import settings

SUMMARIZER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert legal document summarizer.
Convert complex legal findings into clear plain English for non-lawyers.
Be accurate, structured, concise, and highlight key obligations and risks."""
    ),
    (
        "human",
        """Summarise this legal research in plain English.

Original Query: {original_query}

Retrieved Clauses:
{retrieved_clauses}

Clause Classification: {classification}

Risk Analysis:
{risk_analysis}

Format your response as:

## Summary
2-3 sentence overview of what was found.

## Key Points
- Bullet points of the most important obligations and rights

## Risk Assessment
Plain-English explanation of the risk level and main concerns.

## Recommended Actions
What the reader should do or watch out for.
"""
    ),
])


def create_summarizer_chain():
    llm = ChatAnthropic(
        model=settings.llm_model,
        temperature=0,
        anthropic_api_key=settings.anthropic_api_key,
        max_tokens=2048,
    )
    return SUMMARIZER_PROMPT | llm | StrOutputParser()
