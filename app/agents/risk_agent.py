"""
RiskAgent — detects risky language patterns in legal clauses.
Returns structured JSON with risk_level, risk_factors, and recommendation.
"""
from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.config import settings

# NOTE: All { } inside the system prompt that are NOT template variables
# must be escaped as {{ }} otherwise LangChain tries to fill them as variables.

RISK_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a senior legal risk analyst specializing in contract review.

Analyze the given legal clause for these risk indicators:
- Unlimited or uncapped liability exposure
- Unilateral modification rights
- Broad or asymmetric indemnification obligations
- Vague or undefined terms
- Unfavorable termination clauses
- Missing limitation-of-remedies provisions
- Auto-renewal with no opt-out window
- Mandatory arbitration or class-action waivers
- Intellectual property assignment overreach
- Unreasonable non-compete scope

You MUST respond with ONLY a valid JSON object using this exact structure
(no markdown, no code fences, no extra text before or after):

{{
  "risk_level": "low",
  "risk_factors": ["factor one", "factor two"],
  "recommendation": "brief plain-English action to take",
  "flagged_phrases": ["exact risky phrase from the clause"]
}}

The value of risk_level must be exactly one of: low, medium, high
""",
    ),
    ("human", "Clause:\n{clause_text}"),
])


def create_risk_chain():
    llm = ChatAnthropic(
        model=settings.llm_model,
        temperature=0,
        anthropic_api_key=settings.anthropic_api_key,
        max_tokens=1024,
    )
    return RISK_PROMPT | llm | StrOutputParser()
