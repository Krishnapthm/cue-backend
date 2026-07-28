"""`CueContext`: the seam that gets an `AsyncSession` into a node safely.

The graph is typed against it, `run_turn` supplies it per invocation, and
nothing about it is checkpointed - which is the whole point, since a resume
after an `interrupt()` arrives on a new request holding a new session.
"""

from __future__ import annotations

import dataclasses
import uuid
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent import graph as graph_module
from app.agent.context import CueContext
from app.agent.state import AgentState


def _context(session: AsyncSession, address_id: str = "addr-1") -> CueContext:
    return CueContext(
        session=session,
        user_id=7,
        chat_session_id=uuid.uuid4(),
        address_id=address_id,
    )


def _state() -> AgentState:
    return {"session_id": "s", "user_id": 7, "messages": []}


async def test_a_node_reaches_the_session_through_the_runtime(
    db_session: AsyncSession,
) -> None:
    seen: dict[str, Any] = {}

    def node(state: AgentState, runtime: Runtime[CueContext]) -> dict[str, Any]:
        seen["session"] = runtime.context.session
        seen["address_id"] = runtime.context.address_id
        seen["user_id"] = runtime.context.user_id
        return {}

    builder: StateGraph[AgentState, CueContext] = StateGraph(
        AgentState, context_schema=CueContext
    )
    builder.add_node("node", node)
    builder.add_edge(START, "node")
    builder.add_edge("node", END)

    await builder.compile().ainvoke(_state(), context=_context(db_session, "addr-9"))

    # The same session object the request is using, not a second one opened
    # behind its back - so a node's writes join the request's unit of work.
    assert seen["session"] is db_session
    assert seen["address_id"] == "addr-9"
    assert seen["user_id"] == 7


def test_the_real_graph_declares_the_context_schema() -> None:
    assert graph_module.build_graph().context_schema is CueContext


async def test_the_context_is_not_checkpointed_and_can_change_on_resume(
    db_session: AsyncSession,
) -> None:
    # An interrupt ends the invocation; the resume arrives on a *new* request
    # with a *new* session. A per-invocation context gets that right for
    # free, where a session smuggled into state would be stale by now.
    addresses: list[str] = []

    def pausing_node(state: AgentState, runtime: Runtime[CueContext]) -> Command[Any]:
        addresses.append(runtime.context.address_id)
        interrupt("pick one")
        return Command(update={})

    builder: StateGraph[AgentState, CueContext] = StateGraph(
        AgentState, context_schema=CueContext
    )
    builder.add_node("pausing", pausing_node)
    builder.add_edge(START, "pausing")
    builder.add_edge("pausing", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    config = graph_module.thread_config(str(uuid.uuid4()))

    first = await graph.ainvoke(
        _state(), config, context=_context(db_session, "addr-first")
    )
    assert first["__interrupt__"]

    # A checkpointed context would replay "addr-first" here.
    resumed = await graph.ainvoke(
        Command(resume="ok"), config, context=_context(db_session, "addr-second")
    )

    assert "__interrupt__" not in resumed
    assert addresses == ["addr-first", "addr-second"]


def test_the_context_is_frozen(db_session: AsyncSession) -> None:
    # A node that mutated the shared context would leak turn state sideways
    # into every other node in the same invocation.
    with pytest.raises(dataclasses.FrozenInstanceError):
        _context(db_session).address_id = "addr-other"  # type: ignore[misc]
