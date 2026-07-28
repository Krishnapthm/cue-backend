"""State channel behaviour: reducers, and what parallel writers do to them.

These tests run tiny purpose-built graphs over the real `AgentState`. The
point is the *channel*, not any particular node, so nothing here needs a model
or a database.
"""

from __future__ import annotations

from typing import Any, TypedDict

import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from app.agent.context import CueContext
from app.agent.schemas import MatchResult, TurnIntent
from app.agent.state import AgentState
from app.cart.schemas import MatchStatus


def _match(name: str) -> MatchResult:
    return MatchResult(ingredient_name=name, status=MatchStatus.MATCHED)


class _WorkerInput(TypedDict):
    """What one `Send` hands a fan-out worker - a slice, not the whole state."""

    ingredient: str


def _worker(state: _WorkerInput) -> dict[str, Any]:
    return {"matches": [_match(state["ingredient"])]}


def _fan_out(state: AgentState) -> list[Send]:
    return [Send("worker", {"ingredient": name}) for name in ("paneer", "butter")]


def _start_state() -> AgentState:
    return {"session_id": "s", "user_id": 1, "messages": []}


def _fan_out_graph() -> Any:
    builder: StateGraph[AgentState, CueContext] = StateGraph(
        AgentState, context_schema=CueContext
    )
    builder.add_node("worker", _worker)
    builder.add_conditional_edges(START, _fan_out, ["worker"])
    builder.add_edge("worker", END)
    return builder.compile()


async def test_two_simultaneous_writes_to_matches_accumulate() -> None:
    # The `Send` fan-out's whole point is that N workers write `matches` in
    # the same super-step. Without `operator.add` the last one to finish wins
    # and the rest vanish - quietly, as a short checklist rather than an
    # error. This is the test that would fail if the reducer were dropped.
    result = await _fan_out_graph().ainvoke(_start_state())

    assert [m.ingredient_name for m in result["matches"]] == ["paneer", "butter"]


async def test_matches_defaults_to_empty_without_being_passed() -> None:
    # `matches` is NotRequired so pre-existing state literals stay valid; the
    # reducer still supplies a starting value.
    result = await _fan_out_graph().ainvoke(_start_state())

    assert isinstance(result["matches"], list)


async def test_matches_accumulates_across_super_steps_too() -> None:
    builder: StateGraph[AgentState, CueContext] = StateGraph(
        AgentState, context_schema=CueContext
    )
    builder.add_node("first", lambda state: {"matches": [_match("paneer")]})
    builder.add_node("second", lambda state: {"matches": [_match("butter")]})
    builder.add_edge(START, "first")
    builder.add_edge("first", "second")
    builder.add_edge("second", END)

    result = await builder.compile().ainvoke(_start_state())

    assert [m.ingredient_name for m in result["matches"]] == ["paneer", "butter"]


@pytest.mark.parametrize(
    "field",
    ["turn_intent", "matches", "cart_plan_id", "compose_result", "failure"],
)
def test_new_fields_are_all_optional(field: str) -> None:
    # A state literal written before these fields existed - the ones already
    # in this suite, and the one `chat.service` builds - must stay valid.
    # Asserted against the annotation rather than `__required_keys__`: the
    # module uses `from __future__ import annotations`, so every annotation
    # is an unresolved string at runtime and TypedDict cannot classify any
    # key. mypy, which reads them statically, is the enforcing check.
    assert "NotRequired" in str(AgentState.__annotations__[field])


def test_the_cart_fields_are_json_serializable() -> None:
    # Everything in state is checkpointed, so a field that cannot round-trip
    # through the serializer breaks persistence rather than typing.
    row = MatchResult(
        ingredient_name="paneer",
        status=MatchStatus.SUBSTITUTED,
        substitution_reason="Amul 500g out of stock",
    )

    assert MatchResult.model_validate_json(row.model_dump_json()) == row
    assert TurnIntent("recipe") is TurnIntent.RECIPE
