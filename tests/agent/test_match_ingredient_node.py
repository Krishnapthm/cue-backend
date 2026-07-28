"""The ingredient fan-out: N workers, N rows, streamed as they land (CUE-91).

Most of these run against a small harness graph that registers
`match_ingredient` exactly as `build_graph` does - same `input_schema`, same
`retry_policy`, same conditional edge - so the `Send` fan-out, the
`operator.add` reducer and the custom stream are all exercised for real
without dragging a recipe turn's model calls in behind them. The structural
test at the bottom pins the wiring in the real graph to this harness.

No model is called from anywhere in this module: nothing on this path takes
one. `select_variant` and `propose_substitute` are deterministic by design.
"""

from __future__ import annotations

import asyncio
import uuid
from decimal import Decimal
from typing import Any

import pytest
from langgraph.graph import END, START, StateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent import graph as graph_module
from app.agent.context import CueContext
from app.agent.nodes import match_ingredient as node_module
from app.agent.nodes.match_ingredient import (
    MATCH_INGREDIENT,
    MatchTask,
    fan_out,
    match_ingredient,
    needed_ingredients,
)
from app.agent.schemas import IngredientStatus, MatchResult, NormalizedIngredient
from app.agent.state import AgentState
from app.cart.schemas import Ingredient, MatchStatus
from app.chat.schemas import MatchEvent
from app.instamart import service as instamart_service
from app.instamart.exceptions import (
    InstamartAuthError,
    InstamartDomainError,
    InstamartTransportError,
)
from app.instamart.schemas import Product, ProductVariant
from app.matching import substitution as matching_service
from app.matching.schemas import SubstitutionResult
from app.models.user import User
from tests.conftest import InstamartToolCallStub

PANEER = NormalizedIngredient(
    name="paneer", quantity=250, unit="g", status=IngredientStatus.NEED
)
BUTTER = NormalizedIngredient(
    name="butter", quantity=100, unit="g", status=IngredientStatus.NEED
)
CREAM = NormalizedIngredient(
    name="cream", quantity=200, unit="ml", status=IngredientStatus.NEED
)
SALT = NormalizedIngredient(
    name="salt", quantity=None, unit=None, status=IngredientStatus.HAVE
)


def _state(
    rows: list[NormalizedIngredient], have: set[str] | None = None
) -> AgentState:
    state: AgentState = {
        "session_id": "session-1",
        "user_id": 1,
        "messages": [],
        "normalized_ingredients": rows,
        "matches": [],
    }
    if have is not None:
        state["have_marks"] = have
    return state


def _context() -> CueContext:
    return CueContext(
        session=None,  # type: ignore[arg-type]
        user_id=1,
        chat_session_id=uuid.uuid4(),
        address_id="addr-1",
    )


def _harness() -> Any:
    """Compile a graph that fans out and nothing else.

    Registered exactly as `build_graph` registers it; `test_the_real_graph_...`
    below is what keeps the two honest with each other.
    """
    builder: StateGraph[AgentState, CueContext] = StateGraph(
        AgentState, context_schema=CueContext
    )
    builder.add_node(
        MATCH_INGREDIENT,
        match_ingredient,
        input_schema=MatchTask,
        retry_policy=graph_module.NETWORK_RETRY,
    )
    builder.add_conditional_edges(START, fan_out, [MATCH_INGREDIENT, END])
    builder.add_edge(MATCH_INGREDIENT, END)
    return builder.compile()


def _product(
    name: str,
    *,
    spin_id: str,
    pack_size: str,
    price: str,
    in_stock: bool = True,
    brand: str | None = None,
) -> Product:
    return Product(
        product_id=f"prod-{spin_id}",
        name=name,
        brand=brand,
        variants=[
            ProductVariant(
                spin_id=spin_id,
                pack_size=pack_size,
                price=Decimal(price),
                in_stock=in_stock,
            )
        ],
    )


def _stub_search(
    monkeypatch: pytest.MonkeyPatch, by_query: dict[str, Any]
) -> list[str]:
    """Answer `search_products` per query; record the order calls were made in.

    A `by_query` value is a product list, an exception (class or instance) to
    raise, or an async callable - so a single ingredient can be made to fail,
    or to lag, while its siblings succeed.
    """
    seen: list[str] = []

    async def _fake(
        _session: Any, _user_id: int, *, address_id: str, query: str, **_: Any
    ) -> list[Product]:
        seen.append(query)
        answer = by_query.get(query, [])
        if isinstance(answer, type) and issubclass(answer, Exception):
            raise answer
        if isinstance(answer, Exception):
            raise answer
        if callable(answer):
            produced: list[Product] = await answer()
            return produced
        return list(answer)

    monkeypatch.setattr(instamart_service, "search_products", _fake)
    return seen


def _rows(result: dict[str, Any]) -> dict[str, MatchResult]:
    return {match.ingredient_name: match for match in result["matches"]}


# --- what gets fanned out ---------------------------------------------------


def test_fan_out_sends_one_worker_per_needed_ingredient() -> None:
    sends = fan_out(_state([PANEER, BUTTER, SALT], have={"salt"}))

    assert isinstance(sends, list)
    assert [send.node for send in sends] == [MATCH_INGREDIENT, MATCH_INGREDIENT]
    assert [send.arg["ingredient"].name for send in sends] == ["paneer", "butter"]


def test_fan_out_ends_the_turn_when_the_user_already_has_everything() -> None:
    """An empty `Send` list would strand the turn with no next node at all."""
    assert fan_out(_state([SALT], have={"salt"})) == END


def test_the_users_marks_beat_the_stale_pantry_seeded_status() -> None:
    """The checklist answer is what authorizes spending money, not the seed.

    `normalized_ingredients` is built before the interrupt, so its statuses
    are the pantry's guess. After the resume the user has said paneer is
    already in the fridge and salt is not - both the opposite of the seed.
    """
    rows = needed_ingredients(_state([PANEER, SALT], have={"paneer"}))

    assert [row.name for row in rows] == ["salt"]


def test_without_any_marks_the_rows_own_status_still_decides() -> None:
    assert [row.name for row in needed_ingredients(_state([PANEER, SALT]))] == [
        "paneer"
    ]


# --- N ingredients, N rows --------------------------------------------------


async def test_every_ingredient_produces_exactly_one_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The count is the assertion: a missing reducer silently loses N-1 rows."""
    _stub_search(
        monkeypatch,
        {
            "paneer": [_product("Paneer", spin_id="s1", pack_size="200 g", price="90")],
            "butter": [_product("Butter", spin_id="s2", pack_size="100 g", price="55")],
            "cream": [_product("Cream", spin_id="s3", pack_size="200 ml", price="70")],
        },
    )

    result = await _harness().ainvoke(
        _state([PANEER, BUTTER, CREAM]), context=_context()
    )

    assert len(result["matches"]) == 3
    assert set(_rows(result)) == {"paneer", "butter", "cream"}
    assert all(row.status is MatchStatus.MATCHED for row in result["matches"])


async def test_quantity_math_carries_through_to_the_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """250 g needed from 200 g packs is two packs, per `select_variant`."""
    _stub_search(
        monkeypatch,
        {"paneer": [_product("Paneer", spin_id="s1", pack_size="200 g", price="90")]},
    )

    result = await _harness().ainvoke(_state([PANEER]), context=_context())

    row = _rows(result)["paneer"]
    assert row.quantity == 2
    assert row.spin_id == "s1"
    assert row.unit_price == Decimal("90")


# --- three outcomes, all of them visible ------------------------------------


async def test_a_substitution_carries_the_reason_it_was_swapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_search(
        monkeypatch,
        {"paneer": [_product("Paneer", spin_id="s1", pack_size="200 g", price="90")]},
    )

    async def _substitute(*_args: Any, **_kwargs: Any) -> SubstitutionResult:
        return SubstitutionResult(
            spin_id="s9",
            product_name="Nandini Paneer",
            pack_size="250 g",
            unit_price=Decimal("99"),
            quantity=1,
            reason="paneer in 200 g was out of stock, so we substituted Nandini.",
        )

    # Nothing in stock, so `select_variant` gives up and the substitution
    # path is what answers.
    _stub_search(
        monkeypatch,
        {
            "paneer": [
                _product(
                    "Paneer",
                    spin_id="s1",
                    pack_size="200 g",
                    price="90",
                    in_stock=False,
                )
            ]
        },
    )
    monkeypatch.setattr(matching_service, "propose_substitute", _substitute)

    result = await _harness().ainvoke(_state([PANEER]), context=_context())

    row = _rows(result)["paneer"]
    assert row.status is MatchStatus.SUBSTITUTED
    assert row.spin_id == "s9"
    assert row.substitution_reason is not None
    assert "out of stock" in row.substitution_reason


async def test_a_brand_substitution_from_select_variant_keeps_its_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`select_variant` can substitute too, and that reason must survive."""
    _stub_search(
        monkeypatch,
        {
            "paneer": [
                _product(
                    "Paneer",
                    spin_id="s1",
                    pack_size="250 g",
                    price="90",
                    brand="Nandini",
                )
            ]
        },
    )
    monkeypatch.setattr(
        node_module,
        "_as_ingredient",
        lambda row: Ingredient(
            name=row.name,
            quantity=Decimal("250"),
            unit="g",
            preferred_brand="Amul",
        ),
    )

    result = await _harness().ainvoke(_state([PANEER]), context=_context())

    row = _rows(result)["paneer"]
    assert row.status is MatchStatus.SUBSTITUTED
    assert row.substitution_reason is not None
    assert "Amul" in row.substitution_reason


async def test_an_unavailable_ingredient_is_represented_not_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cart that silently omits an item looks complete and is not."""
    _stub_search(monkeypatch, {"paneer": [], "butter": []})

    async def _no_substitute(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(matching_service, "propose_substitute", _no_substitute)

    result = await _harness().ainvoke(_state([PANEER, BUTTER]), context=_context())

    assert len(result["matches"]) == 2
    for row in result["matches"]:
        assert row.status is MatchStatus.UNAVAILABLE
        assert row.spin_id is None
        assert row.substitution_reason is not None


# --- streaming --------------------------------------------------------------


async def test_each_row_streams_on_its_own_keyed_by_ingredient_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_search(
        monkeypatch,
        {
            "paneer": [_product("Paneer", spin_id="s1", pack_size="250 g", price="90")],
            "butter": [_product("Butter", spin_id="s2", pack_size="100 g", price="55")],
            "cream": [_product("Cream", spin_id="s3", pack_size="200 ml", price="70")],
        },
    )

    events = [
        chunk
        async for chunk in _harness().astream(
            _state([PANEER, BUTTER, CREAM]), context=_context(), stream_mode="custom"
        )
    ]

    # One event per ingredient, not one batch at the end of the super-step.
    assert len(events) == 3
    assert {event["ingredient_name"] for event in events} == {
        "paneer",
        "butter",
        "cream",
    }
    # JSON-shaped, because it crosses to a client as an SSE payload.
    assert all(isinstance(event["status"], str) for event in events)


async def test_a_slow_worker_does_not_hold_up_the_others(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Workers finish out of order, and the stream is expected to reflect that.

    Paneer is first in recipe order and slowest to come back. Its event must
    arrive last, which is exactly why `MatchEvent` keys on the ingredient name
    rather than on arrival position - the client fills the row in place.
    """

    async def _slow() -> list[Product]:
        await asyncio.sleep(0.05)
        return [_product("Paneer", spin_id="s1", pack_size="250 g", price="90")]

    _stub_search(
        monkeypatch,
        {
            "paneer": _slow,
            "butter": [_product("Butter", spin_id="s2", pack_size="100 g", price="55")],
        },
    )

    events = [
        chunk
        async for chunk in _harness().astream(
            _state([PANEER, BUTTER]), context=_context(), stream_mode="custom"
        )
    ]

    assert [event["ingredient_name"] for event in events] == ["butter", "paneer"]


async def test_the_emitted_payload_is_exactly_what_the_sse_layer_parses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The seam between this node and CUE-87's `custom` handler.

    `chat.service._custom_event` validates whatever a node writes back into a
    `MatchResult` and drops anything it cannot read - quietly, by design. So a
    drift in what this node emits would not fail a test over there; it would
    just stop rendering rows. This asserts the round trip instead.
    """
    _stub_search(
        monkeypatch,
        {"paneer": [_product("Paneer", spin_id="s1", pack_size="250 g", price="90")]},
    )

    events = [
        chunk
        async for chunk in _harness().astream(
            _state([PANEER]), context=_context(), stream_mode="custom"
        )
    ]

    event = MatchEvent.from_match(MatchResult.model_validate(events[0]))
    assert event.ingredient_name == "paneer"
    assert event.status is MatchStatus.MATCHED
    assert event.unit_price == Decimal("90")


# --- failure isolation ------------------------------------------------------


async def test_one_ingredients_rejection_does_not_fail_the_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Swiggy executed the search and said no: that is about one ingredient."""
    _stub_search(
        monkeypatch,
        {
            "paneer": InstamartDomainError,
            "butter": [_product("Butter", spin_id="s2", pack_size="100 g", price="55")],
            "cream": [_product("Cream", spin_id="s3", pack_size="200 ml", price="70")],
        },
    )

    result = await _harness().ainvoke(
        _state([PANEER, BUTTER, CREAM]), context=_context()
    )

    rows = _rows(result)
    assert len(rows) == 3
    assert rows["paneer"].status is MatchStatus.UNAVAILABLE
    assert rows["butter"].status is MatchStatus.MATCHED
    assert rows["cream"].status is MatchStatus.MATCHED


async def test_auth_expiry_ends_the_turn_rather_than_failing_every_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The token is dead, so every remaining worker would fail identically."""
    _stub_search(
        monkeypatch,
        {
            "paneer": InstamartAuthError,
            "butter": [_product("Butter", spin_id="s2", pack_size="100 g", price="55")],
        },
    )

    with pytest.raises(InstamartAuthError):
        await _harness().ainvoke(_state([PANEER, BUTTER]), context=_context())


async def test_an_outage_is_retried_and_then_ends_the_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transport failures are retried by `NETWORK_RETRY`, not swallowed.

    Degrading them to `unavailable` would compose a cart missing real items
    because Swiggy was down - the user finds that out at the stove.
    """
    seen = _stub_search(monkeypatch, {"paneer": InstamartTransportError})

    with pytest.raises(InstamartTransportError):
        await _harness().ainvoke(_state([PANEER]), context=_context())

    assert seen.count("paneer") == graph_module.NETWORK_RETRY.max_attempts


async def test_a_transient_failure_that_recovers_still_yields_a_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = {"n": 0}

    async def _flaky() -> list[Product]:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise InstamartTransportError
        return [_product("Paneer", spin_id="s1", pack_size="250 g", price="90")]

    _stub_search(monkeypatch, {"paneer": _flaky})

    result = await _harness().ainvoke(_state([PANEER]), context=_context())

    assert _rows(result)["paneer"].status is MatchStatus.MATCHED
    assert attempts["n"] == 2


# --- against the real Swiggy boundary ---------------------------------------


async def test_a_real_search_response_resolves_to_a_purchasable_row(
    db_session: AsyncSession,
    linked_user: User,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    """End to end over the MCP wire shape, mocking only the Swiggy call.

    The rest of this module stubs `search_products`; this one goes through the
    real client and parser, so a change in Swiggy's envelope cannot pass the
    fan-out's tests by agreeing with a fake.
    """
    mock_instamart_tool_call.configure_text_envelope(
        {
            "success": True,
            "data": {
                "products": [
                    {
                        "productId": "prod-1",
                        "name": "Amul Malai Paneer",
                        "brand": "Amul",
                        "variants": [
                            {
                                "spinId": "SPIN1",
                                "packSize": "200 g",
                                "price": "95.00",
                                "inStock": True,
                            }
                        ],
                    }
                ]
            },
        }
    )
    context = CueContext(
        session=db_session,
        user_id=linked_user.id,
        chat_session_id=uuid.uuid4(),
        address_id="addr-1",
    )

    result = await _harness().ainvoke(_state([PANEER]), context=context)

    row = _rows(result)["paneer"]
    assert row.status is MatchStatus.MATCHED
    assert row.spin_id == "SPIN1"
    assert row.product_name == "Amul Malai Paneer"
    assert row.quantity == 2
    assert mock_instamart_tool_call.tool_calls("search_products") == [
        {"addressId": "addr-1", "query": "paneer", "offset": 0}
    ]


# --- the wiring in the real graph -------------------------------------------


def test_the_real_graph_fans_out_of_confirm_checklist() -> None:
    """The harness above is only worth anything if the real graph matches it."""
    compiled = graph_module.build_graph().compile()
    graph = compiled.get_graph()

    assert MATCH_INGREDIENT in compiled.nodes

    out_of_checklist = [
        edge for edge in graph.edges if edge.source == graph_module.CONFIRM_CHECKLIST
    ]
    assert out_of_checklist, "confirm_checklist must still lead somewhere"
    # The fan-out is conditional. A static edge alongside it would run both
    # branches - the same trap `route_turn` documents.
    assert all(edge.conditional for edge in out_of_checklist)
    assert {edge.target for edge in out_of_checklist} == {MATCH_INGREDIENT, END}


def test_the_worker_carries_the_network_retry_policy() -> None:
    node = graph_module.build_graph().nodes[MATCH_INGREDIENT]

    assert node.retry_policy == graph_module.NETWORK_RETRY
