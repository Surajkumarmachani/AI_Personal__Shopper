"""
agent.py
========
Builds a LangChain ReAct agent powered by OpenAI GPT-4o with per-session memory
and exposes one async function, run_agent(), that the FastAPI layer calls.

Architecture:
  - LLM:    ChatOpenAI(model="gpt-4o") from langchain-openai
  - Agent:  LangGraph create_react_agent — handles the ReAct loop (reason → act
            → observe) for tool routing automatically.
  - Memory: InMemorySaver checkpointer keyed by thread_id (= session_id) keeps
            each shopper's conversation history within a session.
  - Emotion: The shopper's detected emotion is injected as a system-level context
            message so the LLM adapts its tone naturally without the shopper
            seeing any trace of it.
"""

import logging
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

from config import (
    LLM_MODEL,
    BASE_SYSTEM_PROMPT,
    PRODUCT_BUFFER,
    EMOTION_TONE_MAP,
    wrap_with_emotion,
)
from product_search import search_products, set_session_id

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Build the agent once, at import time
# ─────────────────────────────────────────────
# InMemorySaver holds conversation state per thread_id. It's in-memory only —
# good for local/dev. For production, swap to a Postgres/SQLite checkpointer so
# history survives restarts and scales across processes.
_checkpointer = InMemorySaver()

# ChatOpenAI wraps the OpenAI API; OPENAI_API_KEY is read from env automatically.
_llm = ChatOpenAI(model=LLM_MODEL, temperature=0.7)

# create_react_agent builds a LangGraph ReAct agent that:
#   1. Sends the user message + system prompt to the LLM
#   2. If the LLM calls a tool, executes it and feeds the result back
#   3. Repeats until the LLM produces a final text response
agent = create_react_agent(
    _llm,
    tools=[search_products],
    prompt=BASE_SYSTEM_PROMPT,
    checkpointer=_checkpointer,
)


def _extract_text(message) -> str:
    """Pull plain text out of the final agent message (handles content blocks)."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            c.get("text", "") if isinstance(c, dict) else str(c) for c in content
        )
    return str(content)


async def run_agent(message: str, emotion: str, session_id: str) -> tuple[str, list[dict]]:
    """
    Run one conversation turn.

    Returns (assistant_reply, products) where `products` is the structured list
    of items the agent searched for this turn (empty if it didn't search).
    """
    # Clear last turn's captured products for this session
    PRODUCT_BUFFER.pop(session_id, None)

    # Set the session_id so the search tool can tag results for this session
    set_session_id(session_id)

    # Inject the current emotion as a tone note (no-op for "neutral").
    # This gets prepended to the user message as invisible context the agent
    # honors but never repeats — effectively system-level emotion awareness.
    wrapped = wrap_with_emotion(message, emotion)

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": wrapped}]},
        config={"configurable": {"thread_id": session_id}},
    )

    reply = _extract_text(result["messages"][-1])
    products = PRODUCT_BUFFER.pop(session_id, [])
    return reply, products
