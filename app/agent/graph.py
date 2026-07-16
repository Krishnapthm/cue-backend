from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agent.observability import configure_tracing
from app.agent.state import AgentState
from app.config import settings

logger = logging.getLogger(__name__)

SMOKE_TEST_MESSAGE = "cue-agent scaffold online"


def smoke_test_node(state: AgentState) -> dict[str, Any]:
    """Prove the scaffold runs end to end.

    Deliberately does no model or network I/O so the graph is invocable (and
    every run traceable) without provider credentials. It emits a single
    marker message; the `add_messages` reducer appends it to the transcript.

    The return is a partial state update (LangGraph's node convention) rather
    than a full `AgentState`, which is why it is typed `dict[str, Any]` instead
    of `AgentState` - a partial dict cannot satisfy the total `AgentState`
    TypedDict under strict typing.
    """
    logger.info("smoke_test_node running for session %s", state["session_id"])
    return {"messages": [AIMessage(content=SMOKE_TEST_MESSAGE)]}


def build_graph() -> StateGraph[AgentState]:
    """Build the agent's (uncompiled) graph.

    Skeleton: `START -> smoke_test_node -> END`. Later issues (recipe
    generation, photo parse, normalization, substitution) add real nodes here
    without touching this scaffold. Tracing is configured as a side effect so
    any `build_graph().compile().ainvoke(...)` path is captured in LangSmith
    when a key is present.

    Returns:
        The uncompiled `StateGraph`; the caller compiles it (optionally with a
        checkpointer via `open_compiled_graph`).
    """
    configure_tracing()

    builder: StateGraph[AgentState] = StateGraph(AgentState)
    builder.add_node("smoke_test_node", smoke_test_node)
    builder.add_edge(START, "smoke_test_node")
    builder.add_edge("smoke_test_node", END)
    return builder


def _checkpointer_conn_string() -> str:
    """Return a psycopg-compatible DSN for the checkpointer.

    `AsyncPostgresSaver` speaks psycopg (psycopg3), not SQLAlchemy's asyncpg
    driver, so the `+asyncpg` suffix is stripped from the shared
    `DATABASE_URL`.
    """
    return str(settings.DATABASE_URL).replace("postgresql+asyncpg", "postgresql")


@asynccontextmanager
async def open_compiled_graph(
    conn_string: str | None = None,
) -> AsyncIterator[CompiledStateGraph[AgentState]]:
    """Yield the compiled graph wired to a Postgres checkpointer.

    The checkpointer persists state against the same database as the app,
    keyed by `thread_id = str(chat_session.id)` supplied per-invocation in the
    run config. Callers use it as an async context manager so the underlying
    connection is closed on exit:

    ```python
    async with open_compiled_graph() as graph:
        await graph.ainvoke(state, {"configurable": {"thread_id": session_id}})
    ```

    Args:
        conn_string: Override DSN (used by tests); defaults to the app database.

    Yields:
        The compiled, checkpointed graph.
    """
    dsn = conn_string or _checkpointer_conn_string()
    async with AsyncPostgresSaver.from_conn_string(dsn) as checkpointer:
        await checkpointer.setup()
        yield build_graph().compile(checkpointer=checkpointer)
