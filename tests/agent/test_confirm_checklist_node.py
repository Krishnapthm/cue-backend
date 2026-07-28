"""`confirm_checklist`: the graph's only interrupt, and its resume.

The pause/resume tests run against a real Postgres checkpointer (the suite's
ephemeral container), because an interrupt without a checkpointer is not a
pause - it is an error. The one that matters most resumes through a *second*
compiled graph, which is what a process restart between the question and the
answer actually looks like.
"""

from __future__ import annotations

import ast
import inspect
import uuid
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from pydantic import ValidationError

from app.agent import graph as graph_module
from app.agent.context import CueContext
from app.agent.nodes import confirm_checklist as node_module
from app.agent.nodes.confirm_checklist import checklist_payload
from app.agent.schemas import (
    GeneratedRecipe,
    IngredientStatus,
    NormalizedIngredient,
    RecipeIngredient,
    TurnClassification,
    TurnIntent,
)
from app.agent.state import AgentState


def _rows() -> list[NormalizedIngredient]:
    return [
        NormalizedIngredient(
            name="paneer", quantity=250, unit="g", status=IngredientStatus.NEED
        ),
        NormalizedIngredient(
            name="salt", quantity=None, unit=None, status=IngredientStatus.HAVE
        ),
    ]


def _state() -> AgentState:
    return {
        "session_id": "session-1",
        "user_id": 1,
        "messages": [],
        "normalized_ingredients": _rows(),
    }


def _context() -> CueContext:
    return CueContext(
        session=None,  # type: ignore[arg-type]
        user_id=1,
        chat_session_id=uuid.uuid4(),
        address_id="addr-1",
    )


# --- the payload ------------------------------------------------------------


def test_the_payload_carries_the_ui_discriminator() -> None:
    # The client routes on an explicit field, not on the payload's shape: the
    # design renders this interrupt inline and a future one may not.
    assert checklist_payload(_state())["ui"] == "checklist"


def test_the_payload_carries_every_row_with_its_pantry_seeded_default() -> None:
    payload = checklist_payload(_state())

    assert payload["items"] == [
        {"name": "paneer", "quantity": 250.0, "unit": "g", "have": False},
        {"name": "salt", "quantity": None, "unit": None, "have": True},
    ]


def test_the_payload_is_json_serializable() -> None:
    # It is checkpointed and sent over SSE; anything else fails at the boundary.
    import json

    json.dumps(checklist_payload(_state()))


def test_an_empty_checklist_still_produces_a_payload() -> None:
    state: AgentState = {"session_id": "s", "user_id": 1, "messages": []}

    assert checklist_payload(state) == {"ui": "checklist", "items": []}


# --- the rule that keeps resume safe ----------------------------------------


def test_interrupt_is_the_first_statement_in_the_node_body() -> None:
    """The rule the whole node depends on, enforced rather than documented.

    On resume LangGraph restarts the node from the top, so anything before
    `interrupt()` runs again - once per resume. A log line is harmless; a DB
    write or a message append is a duplicate every single time.
    """
    source = inspect.getsource(node_module.confirm_checklist)
    body = ast.parse(source).body[0]
    assert isinstance(body, ast.FunctionDef)

    statements = body.body
    # Skip the docstring, which is not executable.
    if isinstance(statements[0], ast.Expr) and isinstance(
        statements[0].value, ast.Constant
    ):
        statements = statements[1:]

    first = statements[0]
    assert isinstance(first, ast.Assign), "the first statement must be the interrupt"
    call = first.value
    assert isinstance(call, ast.Call)
    assert isinstance(call.func, ast.Name)
    assert call.func.id == "interrupt"


def test_the_module_performs_no_writes_before_the_interrupt() -> None:
    # A cheap structural guard against the "just one quick thing before we ask"
    # regression: nothing in this module may touch a session or append_message.
    source = Path(inspect.getfile(node_module)).read_text()
    body = source.split('"""', 2)[-1]
    for forbidden in ("session.add", "append_message", "await "):
        assert forbidden not in body


# --- pause and resume through the real checkpointer -------------------------


class _StubRunnable:
    def __init__(self, results: list[Any]) -> None:
        self.results = list(results)

    async def ainvoke(self, _prompt: list[Any]) -> Any:
        return self.results.pop(0)


class _StubChatModel:
    def __init__(self, runnable: _StubRunnable) -> None:
        self._runnable = runnable

    def with_structured_output(self, _schema: type) -> _StubRunnable:
        return self._runnable


@pytest.fixture
def stub_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the router and recipe models, the pantry read, and Swiggy search.

    Search and cart composition are stubbed because a *resumed* turn no longer
    stops at `confirm_checklist`: it carries on into the ingredient fan-out
    (CUE-91) and then into `compose_cart` (CUE-92), both of which reach off-box
    and the second of which writes. Answering with no candidates and a canned
    compose result keeps these tests about pause and resume.
    """
    from decimal import Decimal

    from app.agent.nodes import recipe as recipe_module
    from app.agent.nodes import route as route_module
    from app.cart import service as cart_service
    from app.cart.schemas import ComposeCartResult
    from app.instamart import service as instamart_service
    from app.instamart.schemas import Product
    from app.pantry import service as pantry_service

    router = _StubRunnable([TurnClassification(intent=TurnIntent.RECIPE, reason="ok")])
    recipe = _StubRunnable(
        [
            GeneratedRecipe(
                dish_name="paneer butter masala",
                estimated_time_minutes=35,
                ingredients=[
                    RecipeIngredient(name="paneer", quantity=250, unit="g"),
                    RecipeIngredient(name="salt"),
                ],
                method_summary="Simmer, fold in paneer.",
            )
        ]
    )
    monkeypatch.setattr(
        route_module, "get_chat_model", lambda _role: _StubChatModel(router)
    )
    monkeypatch.setattr(
        recipe_module, "get_chat_model", lambda _role: _StubChatModel(recipe)
    )

    async def _no_pantry(_session: object, _user_id: int) -> set[str]:
        return set()

    monkeypatch.setattr(pantry_service, "stocked_names", _no_pantry)

    async def _no_candidates(*_args: object, **_kwargs: object) -> list[Product]:
        return []

    # Patched on the service module itself, so `propose_substitute`'s own
    # search is answered by the same stub rather than reaching for a session.
    monkeypatch.setattr(instamart_service, "search_products", _no_candidates)

    async def _compose(*_args: object, **_kwargs: object) -> ComposeCartResult:
        return ComposeCartResult(
            plan_id=1,
            subtotal=Decimal("0"),
            minimum_order_value=Decimal("99"),
            below_minimum=True,
            shortfall=Decimal("99"),
        )

    monkeypatch.setattr(cart_service, "compose_cart", _compose)


def _turn(thread_id: str) -> AgentState:
    return {
        "session_id": thread_id,
        "user_id": 1,
        "messages": [HumanMessage(content="paneer butter masala")],
    }


async def test_a_recipe_turn_pauses_on_the_checklist(
    postgres_url: str, stub_turn: None
) -> None:
    dsn = postgres_url.replace("postgresql+asyncpg", "postgresql")
    thread_id = str(uuid.uuid4())
    config: RunnableConfig = graph_module.thread_config(thread_id)

    async with graph_module.open_compiled_graph(conn_string=dsn) as graph:
        result = await graph.ainvoke(_turn(thread_id), config, context=_context())

        interrupts = result["__interrupt__"]
        assert len(interrupts) == 1
        assert interrupts[0].value["ui"] == "checklist"
        assert [item["name"] for item in interrupts[0].value["items"]] == [
            "paneer",
            "salt",
        ]


async def test_resuming_after_a_process_restart_lands_the_marks(
    postgres_url: str, stub_turn: None
) -> None:
    """The checkpointer test that matters.

    The answer arrives on a *second* compiled graph with its own connection -
    which is what a restart, a redeploy, or simply a different worker looks
    like. Only the `thread_id` connects the two.
    """
    dsn = postgres_url.replace("postgresql+asyncpg", "postgresql")
    thread_id = str(uuid.uuid4())
    config: RunnableConfig = graph_module.thread_config(thread_id)

    async with graph_module.open_compiled_graph(conn_string=dsn) as graph:
        paused = await graph.ainvoke(_turn(thread_id), config, context=_context())
        assert paused["__interrupt__"]

    async with graph_module.open_compiled_graph(conn_string=dsn) as graph:
        resumed = await graph.ainvoke(
            Command(resume={"have": ["salt"]}), config, context=_context()
        )

    assert resumed["have_marks"] == {"salt"}
    assert "__interrupt__" not in resumed or not resumed["__interrupt__"]
    # The resume runs on past the pause into the fan-out, and only for what the
    # user did *not* tick: salt is theirs already, paneer is the shopping list.
    assert [match.ingredient_name for match in resumed["matches"]] == ["paneer"]


async def test_a_resume_on_a_different_thread_does_not_continue_the_first(
    postgres_url: str, stub_turn: None
) -> None:
    """`thread_id` is the only thing tying an answer to its question.

    This is the trap the HITL skill names: a `Command(resume=...)` on a thread
    with no pending interrupt does not error *as a resume* - it starts a fresh
    run from START. Here that fresh run dies immediately because it has no user
    turn to classify, which is the visible symptom; what it never does is apply
    the marks to the session that is actually waiting.

    `chat.service` is what turns this into a loud 409 rather than a mystery, by
    reading the checkpointer before it decides how to invoke.
    """
    dsn = postgres_url.replace("postgresql+asyncpg", "postgresql")
    paused_thread = str(uuid.uuid4())
    other_thread = str(uuid.uuid4())

    async with graph_module.open_compiled_graph(conn_string=dsn) as graph:
        await graph.ainvoke(
            _turn(paused_thread),
            graph_module.thread_config(paused_thread),
            context=_context(),
        )

        with pytest.raises(ValueError, match="no messages"):
            await graph.ainvoke(
                Command(resume={"have": ["salt"]}),
                graph_module.thread_config(other_thread),
                context=_context(),
            )

        still_paused = await graph.aget_state(graph_module.thread_config(paused_thread))

    assert still_paused.interrupts, "the real session must still be waiting"
    assert not still_paused.values.get("have_marks")


async def test_an_unreadable_resume_is_rejected(
    postgres_url: str, stub_turn: None
) -> None:
    # A resume we cannot read is not consent. Coercing it to "none of them"
    # would silently buy the user everything they already own.
    dsn = postgres_url.replace("postgresql+asyncpg", "postgresql")
    thread_id = str(uuid.uuid4())
    config: RunnableConfig = graph_module.thread_config(thread_id)

    async with graph_module.open_compiled_graph(conn_string=dsn) as graph:
        await graph.ainvoke(_turn(thread_id), config, context=_context())

        with pytest.raises(ValidationError):
            await graph.ainvoke(
                Command(resume={"ticked": ["salt"]}), config, context=_context()
            )
