"""`compose_cart` and `report_cart`: the plan, the push, and the card (CUE-92).

`compose_cart_node` runs against the real `cart.service.compose_cart` and a
real ephemeral Postgres, mocking only the Swiggy MCP call - the node's whole
job is marshalling, and marshalling is exactly what a stubbed service would
stop testing. `report_cart_node` is pure and needs neither.

No model is called: neither of these nodes takes one.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent import graph as graph_module
from app.agent.context import CueContext
from app.agent.nodes.cart import (
    UNRECORDED_SELECTION_REASON,
    compose_cart_node,
    report_cart_node,
)
from app.agent.schemas import (
    IngredientStatus,
    MatchResult,
    NormalizedIngredient,
)
from app.agent.state import AgentState
from app.cart.schemas import ComposeCartResult, MatchStatus
from app.instamart.constants import TOOL_UPDATE_CART
from app.instamart.schemas import Cart, CartLineItem
from app.models.cart import CartPlan, CartPlanItem
from app.models.chat import ChatSession
from app.models.user import User
from tests.conftest import InstamartToolCallStub

PANEER_ROW = NormalizedIngredient(
    name="paneer", quantity=250, unit="g", status=IngredientStatus.NEED
)
BUTTER_ROW = NormalizedIngredient(
    name="butter", quantity=100, unit="g", status=IngredientStatus.NEED
)


def _match(
    name: str = "paneer",
    *,
    status: MatchStatus = MatchStatus.MATCHED,
    spin_id: str | None = "spin-1",
    quantity: int | None = 2,
    unit_price: str | None = "90.00",
    substitution_reason: str | None = None,
    selection_reason: str | None = "Selected Amul 200 g.",
) -> MatchResult:
    return MatchResult(
        ingredient_name=name,
        status=status,
        spin_id=spin_id,
        product_name="Amul Paneer",
        pack_size="200 g",
        unit_price=Decimal(unit_price) if unit_price is not None else None,
        quantity=quantity,
        substitution_reason=substitution_reason,
        selection_reason=selection_reason,
    )


def _state(
    matches: list[MatchResult],
    rows: list[NormalizedIngredient] | None = None,
    compose_result: ComposeCartResult | None = None,
) -> AgentState:
    state: AgentState = {
        "session_id": "session-1",
        "user_id": 1,
        "messages": [],
        "matches": matches,
        "normalized_ingredients": rows if rows is not None else [PANEER_ROW],
    }
    if compose_result is not None:
        state["compose_result"] = compose_result
    return state


class _Runtime:
    """Minimal stand-in for `Runtime[CueContext]` - nodes only read `.context`."""

    def __init__(self, context: CueContext) -> None:
        self.context = context


@pytest_asyncio.fixture
async def chat_session(db_session: AsyncSession, linked_user: User) -> ChatSession:
    """A persisted chat session, since `cart_plan.session_id` is a real FK."""
    session = ChatSession(user_id=linked_user.id)
    db_session.add(session)
    await db_session.commit()
    await db_session.refresh(session)
    return session


def _context(db_session: AsyncSession, user: User, session: ChatSession) -> Any:
    return _Runtime(
        CueContext(
            session=db_session,
            user_id=user.id,
            chat_session_id=session.id,
            address_id="addr-1",
        )
    )


CART_WITH_BOTH = {
    "items": [
        {"spinId": "spin-1", "quantity": 2, "price": "180.00"},
        {"spinId": "spin-2", "quantity": 1, "price": "55.00"},
    ],
    "total": "235.00",
    "minimumOrderValue": "99.00",
}


# --- compose_cart -----------------------------------------------------------


async def test_compose_persists_the_recipes_quantity_not_the_packs(
    db_session: AsyncSession,
    linked_user: User,
    chat_session: ChatSession,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    """`ingredient_qty` is what the recipe asked for; `quantity` is packs bought.

    They are different numbers - 250 g needed, 2 packs bought - and the plan
    row keeps both. Getting this backwards would make the plan unreadable as an
    audit trail.
    """
    mock_instamart_tool_call.configure(result={"structuredContent": CART_WITH_BOTH})

    update = await compose_cart_node(
        _state([_match()]), _context(db_session, linked_user, chat_session)
    )

    items = (
        (
            await db_session.execute(
                select(CartPlanItem).where(
                    CartPlanItem.plan_id == update["cart_plan_id"]
                )
            )
        )
        .scalars()
        .all()
    )
    assert len(items) == 1
    assert items[0].ingredient_name == "paneer"
    assert items[0].ingredient_qty == Decimal("250")
    assert items[0].ingredient_unit == "g"
    assert items[0].quantity == 2
    assert items[0].selection_reason == "Selected Amul 200 g."


async def test_compose_keeps_a_substitutions_reason(
    db_session: AsyncSession,
    linked_user: User,
    chat_session: ChatSession,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    """`cart_plan_item` has a CHECK that a substitution is never silent."""
    mock_instamart_tool_call.configure(result={"structuredContent": CART_WITH_BOTH})
    substituted = _match(
        status=MatchStatus.SUBSTITUTED,
        substitution_reason="Amul was out of stock.",
        selection_reason="Amul was out of stock.",
    )

    update = await compose_cart_node(
        _state([substituted]), _context(db_session, linked_user, chat_session)
    )

    item = (
        (
            await db_session.execute(
                select(CartPlanItem).where(
                    CartPlanItem.plan_id == update["cart_plan_id"]
                )
            )
        )
        .scalars()
        .one()
    )
    assert item.match_status == "substituted"
    assert item.selection_reason == "Amul was out of stock."


async def test_compose_records_a_row_with_no_reason_rather_than_nothing(
    db_session: AsyncSession,
    linked_user: User,
    chat_session: ChatSession,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure(result={"structuredContent": CART_WITH_BOTH})

    update = await compose_cart_node(
        _state([_match(selection_reason=None)]),
        _context(db_session, linked_user, chat_session),
    )

    item = (
        (
            await db_session.execute(
                select(CartPlanItem).where(
                    CartPlanItem.plan_id == update["cart_plan_id"]
                )
            )
        )
        .scalars()
        .one()
    )
    assert item.selection_reason == UNRECORDED_SELECTION_REASON


async def test_compose_supersedes_the_sessions_previous_live_plan(
    db_session: AsyncSession,
    linked_user: User,
    chat_session: ChatSession,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    """R3.3 is a database guarantee: at most one live plan per session."""
    mock_instamart_tool_call.configure(result={"structuredContent": CART_WITH_BOTH})
    runtime = _context(db_session, linked_user, chat_session)

    first = await compose_cart_node(_state([_match()]), runtime)
    second = await compose_cart_node(_state([_match()]), runtime)

    superseded = await db_session.get(CartPlan, first["cart_plan_id"])
    live = await db_session.get(CartPlan, second["cart_plan_id"])
    assert superseded is not None and superseded.superseded_at is not None
    assert live is not None and live.superseded_at is None


async def test_an_unavailable_row_is_planned_but_never_sent_to_swiggy(
    db_session: AsyncSession,
    linked_user: User,
    chat_session: ChatSession,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    """It is the match-rate denominator, not a cart line."""
    mock_instamart_tool_call.configure(result={"structuredContent": CART_WITH_BOTH})
    matches = [
        _match(),
        _match(
            "butter",
            status=MatchStatus.UNAVAILABLE,
            spin_id=None,
            quantity=None,
            unit_price=None,
            selection_reason="Nothing in stock.",
        ),
    ]

    update = await compose_cart_node(
        _state(matches, rows=[PANEER_ROW, BUTTER_ROW]),
        _context(db_session, linked_user, chat_session),
    )

    items = (
        (
            await db_session.execute(
                select(CartPlanItem).where(
                    CartPlanItem.plan_id == update["cart_plan_id"]
                )
            )
        )
        .scalars()
        .all()
    )
    assert {item.ingredient_name for item in items} == {"paneer", "butter"}

    (sent,) = mock_instamart_tool_call.tool_calls(TOOL_UPDATE_CART)
    assert [item["spinId"] for item in sent["items"]] == ["spin-1"]


async def test_below_the_minimum_the_plan_is_kept_and_nothing_is_pushed(
    db_session: AsyncSession,
    linked_user: User,
    chat_session: ChatSession,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    """There is nothing checkout-able yet, so the user's real cart is untouched."""
    mock_instamart_tool_call.configure(result={"structuredContent": CART_WITH_BOTH})
    cheap = _match(quantity=1, unit_price="10.00")

    update = await compose_cart_node(
        _state([cheap]), _context(db_session, linked_user, chat_session)
    )

    result = update["compose_result"]
    assert result.below_minimum is True
    assert result.shortfall == Decimal("89.00")
    assert await db_session.get(CartPlan, update["cart_plan_id"]) is not None
    assert mock_instamart_tool_call.tool_calls(TOOL_UPDATE_CART) == []


# --- report_cart ------------------------------------------------------------


def _composed(
    *,
    below_minimum: bool = False,
    cart: Cart | None = None,
    subtotal: str = "235.00",
    shortfall: str = "0",
) -> ComposeCartResult:
    return ComposeCartResult(
        plan_id=7,
        subtotal=Decimal(subtotal),
        minimum_order_value=Decimal("99.00"),
        below_minimum=below_minimum,
        shortfall=Decimal(shortfall),
        cart=cart,
    )


def test_the_card_is_rendered_from_the_cart_swiggy_read_back() -> None:
    """Swiggy can answer `success: true` and quietly omit a line it dropped.

    Rendering from what we sent would show that line as bought. The user finds
    out at the stove; the read-back is the only thing that catches it.
    """
    cart = Cart(
        items=[CartLineItem(spin_id="spin-1", quantity=2, price=Decimal("180.00"))],
        total=Decimal("180.00"),
    )
    matches = [_match(), _match("butter", spin_id="spin-2", quantity=1)]

    update = report_cart_node(
        _state(
            matches, rows=[PANEER_ROW, BUTTER_ROW], compose_result=_composed(cart=cart)
        )
    )

    rows = {item.ingredient_name: item for item in update["cart_report"].items}
    assert rows["paneer"].in_cart is True
    assert rows["paneer"].line_total == Decimal("180.00")
    # Sent, accepted, and silently absent from the cart Swiggy returned.
    assert rows["butter"].in_cart is False
    assert rows["butter"].line_total is None


def test_the_quantity_shown_is_the_carts_not_the_one_we_asked_for() -> None:
    cart = Cart(
        items=[CartLineItem(spin_id="spin-1", quantity=1, price=Decimal("90.00"))]
    )

    update = report_cart_node(
        _state([_match(quantity=2)], compose_result=_composed(cart=cart))
    )

    assert update["cart_report"].items[0].quantity == 1


def test_below_the_minimum_the_card_states_the_shortfall() -> None:
    update = report_cart_node(
        _state(
            [_match(quantity=1, unit_price="10.00")],
            compose_result=_composed(
                below_minimum=True, subtotal="10.00", shortfall="89.00"
            ),
        )
    )

    report = update["cart_report"]
    assert report.below_minimum is True
    assert report.shortfall == Decimal("89.00")
    assert "89.00" in report.summary
    assert "cart screen" in report.summary


def test_a_substitution_reaches_the_card_with_its_reason() -> None:
    cart = Cart(
        items=[CartLineItem(spin_id="spin-1", quantity=2, price=Decimal("180.00"))],
        total=Decimal("180.00"),
    )
    substituted = _match(
        status=MatchStatus.SUBSTITUTED, substitution_reason="Amul was out of stock."
    )

    update = report_cart_node(
        _state([substituted], compose_result=_composed(cart=cart))
    )

    report = update["cart_report"]
    assert report.items[0].substitution_reason == "Amul was out of stock."
    assert "1 substituted" in report.summary


def test_an_unavailable_row_is_on_the_card_and_in_the_summary() -> None:
    cart = Cart(items=[], total=Decimal("0"))
    unavailable = _match(
        status=MatchStatus.UNAVAILABLE, spin_id=None, quantity=None, unit_price=None
    )

    update = report_cart_node(
        _state([unavailable], compose_result=_composed(cart=cart))
    )

    report = update["cart_report"]
    assert report.items[0].status is MatchStatus.UNAVAILABLE
    assert report.items[0].in_cart is False
    assert "1 unavailable" in report.summary


def test_a_turn_with_nothing_to_buy_says_so_plainly() -> None:
    """Not "₹0.00, ₹99.00 under the minimum", which is technically true and useless."""
    update = report_cart_node(
        _state([], rows=[], compose_result=_composed(below_minimum=True, subtotal="0"))
    )

    assert "already have everything" in update["cart_report"].summary


def test_report_cart_refuses_to_render_without_a_compose_result() -> None:
    """A missing result means the graph's edges changed under this node."""
    with pytest.raises(RuntimeError, match="no compose_cart result"):
        report_cart_node(_state([_match()]))


# --- the wiring -------------------------------------------------------------


def test_the_turn_ends_at_report_cart() -> None:
    edges = {
        (edge.source, edge.target)
        for edge in graph_module.build_graph().compile().get_graph().edges
    }

    assert ("match_ingredient", "compose_cart") in edges
    assert ("compose_cart", "report_cart") in edges
    assert ("report_cart", "__end__") in edges


def test_the_graph_has_no_checkout_node() -> None:
    """Checkout is a REST call from the cart screen (CUE-80), and stays there.

    No pause in front of a non-idempotent mutation means the double-charge
    failure mode is structurally absent rather than carefully defended against.
    """
    nodes = set(graph_module.build_graph().nodes)

    assert not [name for name in nodes if "checkout" in name or name == "order"]


def test_compose_cart_is_not_retried() -> None:
    """It writes: a retry would supersede its own plan and insert another."""
    node = graph_module.build_graph().nodes["compose_cart"]

    assert node.retry_policy is None
