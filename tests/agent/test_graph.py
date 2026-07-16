"""Graph scaffold: node behaviour, compile+invoke, and checkpointer persistence.

The checkpointer test runs against the same ephemeral Postgres as the rest of
the suite (via the shared `postgres_url` fixture); `AsyncPostgresSaver.setup()`
provisions its own checkpoint tables there.
"""

from __future__ import annotations

import uuid

from langchain_core.runnables import RunnableConfig

from app.agent.graph import (
    SMOKE_TEST_MESSAGE,
    build_graph,
    open_compiled_graph,
    smoke_test_node,
)
from app.agent.state import AgentState


def _initial_state(session_id: str = "session-1", user_id: int = 1) -> AgentState:
    return {"session_id": session_id, "user_id": user_id, "messages": []}


def test_smoke_test_node_appends_marker_message() -> None:
    update = smoke_test_node(_initial_state())

    messages = update["messages"]
    assert len(messages) == 1
    assert messages[0].content == SMOKE_TEST_MESSAGE


async def test_build_graph_compiles_and_runs_end_to_end() -> None:
    graph = build_graph().compile()

    result = await graph.ainvoke(_initial_state())

    # State flows through: the smoke node's message lands on the transcript and
    # the passthrough fields are preserved.
    assert result["session_id"] == "session-1"
    assert result["user_id"] == 1
    assert [m.content for m in result["messages"]] == [SMOKE_TEST_MESSAGE]


async def test_checkpointer_persists_state_across_invocations(
    postgres_url: str,
) -> None:
    # `postgres_url` is the asyncpg DSN the suite migrates against; the
    # checkpointer speaks psycopg, so hand it the stripped form.
    dsn = postgres_url.replace("postgresql+asyncpg", "postgresql")
    thread_id = str(uuid.uuid4())
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    async with open_compiled_graph(conn_string=dsn) as graph:
        first = await graph.ainvoke(_initial_state(thread_id), config)
        assert len(first["messages"]) == 1

        # Re-invoking on the same thread replays the checkpointed transcript;
        # the add_messages reducer appends rather than overwrites, so it grows.
        second = await graph.ainvoke(_initial_state(thread_id), config)
        assert len(second["messages"]) == 2

        # The persisted state is queryable back out of the checkpointer.
        snapshot = await graph.aget_state(config)
        assert len(snapshot.values["messages"]) == 2

        # A different thread starts from a clean slate.
        other_config: RunnableConfig = {
            "configurable": {"thread_id": str(uuid.uuid4())}
        }
        other = await graph.ainvoke(_initial_state("other"), other_config)
        assert len(other["messages"]) == 1
