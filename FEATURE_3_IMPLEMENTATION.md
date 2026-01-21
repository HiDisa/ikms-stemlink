# IKMS Multi-Agent RAG with Context Critic

## Feature 3: Context Critic & Reranker Agent

**Student Name:** Disara Ransini
**Bootcamp:** STEM Link AI Engineer Bootcamp
**Submission Date:** January 21, 2026

---

## Overview

This project implements **Feature 3: Context Critic & Reranker Agent** for the IKMS Multi-Agent RAG system. The Context Critic intelligently filters retrieved document chunks before answer generation, improving accuracy and reducing noise.

---

## What Was Built

### Core Feature: Context Critic Agent

An intelligent agent that sits between **Retrieval** and **Summarization** in the multi-agent pipeline:
```
START → Retrieval → Context Critic → Summarization → Verification → END

New - Context Critic
```

**The Context Critic:**
- Analyzes each retrieved chunk for relevance to the user's question
- Assigns relevance scores: HIGHLY RELEVANT, MARGINAL, IRRELEVANT
- Filters out irrelevant chunks before they reach the answer generation stage
- Provides transparent rationales for each decision

---

## Technical Implementation

### 1. State Schema Updates (`state.py`)

Added new fields to track critic analysis:
```python
class QAState(TypedDict):
    question: str
    context: str | None
    raw_context: str | None  # NEW: Original unfiltered context
    context_rationale: str | None  # NEW: Critic's analysis
    draft_answer: str | None
    answer: str | None
```

### 2. Context Critic Agent (`agents.py`)

Created a new agent with specialized system prompt for relevance scoring:
```python
context_critic_agent = create_agent(
    model=create_chat_model(),
    tools=[],
    system_prompt=CONTEXT_CRITIC_SYSTEM_PROMPT,
)

def context_critic_node(state: QAState) -> QAState:
    # Analyzes chunks and returns filtered context
    ...
```

### 3. Graph Integration (`graph.py`)

Updated the pipeline to include the critic:
```python
builder.add_node("context_critic", context_critic_node)
builder.add_edge("retrieval", "context_critic")
builder.add_edge("context_critic", "summarization")
```

### 4. User Interface (`frontend/index.html`)

Built a clean, responsive web interface that shows:
- Question input
- Real-time statistics (Retrieved/Kept/Filtered)
- Detailed chunk analysis with color-coded relevance
- Final answer

---

## How to Run

### Prerequisites

- Python 3.12+
- OpenAI API key
- Pinecone account and index (1536 dimensions)

### Installation

1. Clone the repository
2. Create virtual environment:
```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
```
3. Install dependencies:
```bash
   pip install -e .
```
4. Configure `.env`:
```env
   OPENAI_API_KEY=your-key
   OPENAI_EMBEDDING_MODEL_NAME=text-embedding-ada-002
   PINECONE_API_KEY=your-key
   PINECONE_INDEX_NAME=ikms-index
```

### Running the Application

1. Start the backend:
```bash
   uvicorn src.app.api:app --reload
```

2. Open `frontend/index.html` in your browser

3. Upload a PDF at http://127.0.0.1:8000/docs (POST /index-pdf)

4. Ask questions through the UI!

---

## Example Usage

**Question:** "What is sketching in the context of vector databases?"

**Result:**
- Retrieved: 4 chunks
- Kept: 2 HIGHLY RELEVANT chunks
- Filtered: 2 IRRELEVANT chunks
- Answer: Accurate, concise explanation based only on relevant content

---

## Key Benefits

1. **Improved Accuracy** - Only relevant chunks reach the answer generation
2. **Reduced Noise** - Irrelevant information is filtered out
3. **Transparency** - Users can see exactly why chunks were kept or removed
4. **Efficiency** - Less token usage in downstream agents

---

## Files Modified/Created

### Modified:
- `src/app/core/agents/state.py` - Added critic state fields
- `src/app/core/agents/agents.py` - Added critic agent and node
- `src/app/core/agents/graph.py` - Integrated critic into pipeline
- `src/app/core/agents/prompts.py` - Added critic system prompt
- `src/app/api.py` - Added CORS and context_rationale to response
- `src/app/models.py` - Updated QAResponse model

### Created:
- `frontend/index.html` - User interface for testing

---

## Future Enhancements

- Adaptive filtering thresholds based on question complexity
- Integration with citation system (Feature 4)
- A/B testing to measure impact on answer quality
- Support for multi-lingual filtering

---

## Conclusion

The Context Critic agent successfully improves the IKMS system by intelligently filtering retrieved content before answer generation. The implementation demonstrates understanding of multi-agent coordination, prompt engineering, and state management in LangGraph.