"""Agent implementations for the multi-agent RAG flow.

This module defines three LangChain agents (Retrieval, Summarization,
Verification) and thin node functions that LangGraph uses to invoke them.
"""

from typing import List

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ..llm.factory import create_chat_model
from .prompts import (
    RETRIEVAL_SYSTEM_PROMPT,
    SUMMARIZATION_SYSTEM_PROMPT,
    VERIFICATION_SYSTEM_PROMPT,
    CONTEXT_CRITIC_SYSTEM_PROMPT,
)
from .state import QAState
from .tools import retrieval_tool


def _extract_last_ai_content(messages: List[object]) -> str:
    """Extract the content of the last AIMessage in a messages list."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return str(msg.content)
    return ""


# Define agents at module level for reuse
retrieval_agent = create_agent(
    model=create_chat_model(),
    tools=[retrieval_tool],
    system_prompt=RETRIEVAL_SYSTEM_PROMPT,
)

context_critic_agent = create_agent(
    model=create_chat_model(),
    tools=[],
    system_prompt=CONTEXT_CRITIC_SYSTEM_PROMPT,
)

summarization_agent = create_agent(
    model=create_chat_model(),
    tools=[],
    system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
)

verification_agent = create_agent(
    model=create_chat_model(),
    tools=[],
    system_prompt=VERIFICATION_SYSTEM_PROMPT,
)


def retrieval_node(state: QAState) -> QAState:
    """Retrieval Agent node: gathers context from vector store.

    This node:
    - Sends the user's question to the Retrieval Agent.
    - The agent uses the attached retrieval tool to fetch document chunks.
    - Extracts the tool's content (CONTEXT string) from the ToolMessage.
    - Stores the consolidated context string in `state["context"]`.
    """
    question = state["question"]

    result = retrieval_agent.invoke({"messages": [HumanMessage(content=question)]})

    messages = result.get("messages", [])
    context = ""

    # Prefer the last ToolMessage content (from retrieval_tool)
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            context = str(msg.content)
            break

    return {
        "context": context,
        "raw_context": context,
    }

def context_critic_node(state: QAState) -> QAState:
    """Context Critic Agent node: filters and ranks retrieved chunks.

    This node:
    - Receives the question and raw retrieved context.
    - Sends both to the Context Critic Agent.
    - Agent analyzes each chunk for relevance.
    - Extracts the FILTERED CONTEXT from the critic's response.
    - Stores the full analysis in `context_rationale`.
    - Updates `context` with only the highly relevant chunks.
    """
    question = state["question"]
    raw_context = state.get("raw_context", "")

    if not raw_context or raw_context.strip() == "":
        return {
            "context": raw_context,
            "context_rationale": "No context to filter",
        }

    user_content = f"""Question: {question}

Retrieved Chunks:
{raw_context}

Please analyze each chunk and provide filtered context."""

    result = context_critic_agent.invoke(
        {"messages": [HumanMessage(content=user_content)]}
    )
    messages = result.get("messages", [])
    critic_response = _extract_last_ai_content(messages)

    filtered_context = raw_context 
    
    if "=== FILTERED CONTEXT ===" in critic_response:
        parts = critic_response.split("=== FILTERED CONTEXT ===")
        if len(parts) > 1:
            filtered_context = parts[1].strip()

    return {
        "context": filtered_context,
        "context_rationale": critic_response,
    }

def summarization_node(state: QAState) -> QAState:
    """Summarization Agent node: generates draft answer from context.

    This node:
    - Sends question + context to the Summarization Agent.
    - Agent responds with a draft answer grounded only in the context.
    - Stores the draft answer in `state["draft_answer"]`.
    """
    question = state["question"]
    context = state.get("context")

    user_content = f"Question: {question}\n\nContext:\n{context}"

    result = summarization_agent.invoke(
        {"messages": [HumanMessage(content=user_content)]}
    )
    messages = result.get("messages", [])
    draft_answer = _extract_last_ai_content(messages)

    return {
        "draft_answer": draft_answer,
    }


def verification_node(state: QAState) -> QAState:
    """Verification Agent node: verifies and corrects the draft answer.

    This node:
    - Sends question + context + draft_answer to the Verification Agent.
    - Agent checks for hallucinations and unsupported claims.
    - Stores the final verified answer in `state["answer"]`.
    """
    question = state["question"]
    context = state.get("context", "")
    draft_answer = state.get("draft_answer", "")

    user_content = f"""Question: {question}

Context:
{context}

Draft Answer:
{draft_answer}

Please verify and correct the draft answer, removing any unsupported claims."""

    result = verification_agent.invoke(
        {"messages": [HumanMessage(content=user_content)]}
    )
    messages = result.get("messages", [])
    answer = _extract_last_ai_content(messages)

    return {
        "answer": answer,
    }
