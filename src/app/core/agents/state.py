"""LangGraph state schema for the multi-agent QA flow."""

from typing import TypedDict


class QAState(TypedDict):
    """State schema for the linear multi-agent QA flow.

    The state flows through four agents:
    1. Retrieval Agent: populates `context` from `question`
    2. Context Critic Agent: filters `context` and creates `context_rationale` 🆕 NEW!
    3. Summarization Agent: generates `draft_answer` from `question` + filtered `context`
    4. Verification Agent: produces final `answer` from `question` + `context` + `draft_answer`
    """

    question: str
    context: str | None
    raw_context: str | None  # 🆕 NEW: Stores original unfiltered context
    context_rationale: str | None  # 🆕 NEW: Critic's reasoning for filtering
    draft_answer: str | None
    answer: str | None