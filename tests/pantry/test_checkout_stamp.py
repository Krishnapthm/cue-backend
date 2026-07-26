"""The last_bought_at stamp, exercised through the real checkout path.

Placing an order is what stamps the pantry, so these drive `place_order`
rather than calling `stamp_last_bought` directly - that is the behaviour the
acceptance criteria describe, and the wiring between the two is exactly what
could break.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.cart.checkout import place_order
from app.models.cart import CartPlan, CartPlanItem
from app.models.chat import ChatSession
from app.models.user import User
from app.pantry import service
from app.pantry.constants import PantryCategory
from app.pantry.schemas import PantryItemCreate
from tests.conftest import InstamartToolCallStub

NON_EMPTY_CART = {
    "items": [{"spinId": "spin-1", "quantity": 2, "price": "60.00"}],
    "total": "120.00",
    "availablePaymentMethods": ["COD"],
}


@pytest.fixture
def placed_checkout(mock_instamart_tool_call: InstamartToolCallStub) -> None:
    """Stub Swiggy so `place_order` reaches the `placed` branch.

    The Swiggy order id is unique per test: the test database is shared for
    the whole session and never truncated, and `uq_order_swiggy_order_id`
    would otherwise reject the second test that places an order.
    """
    mock_instamart_tool_call.configure_tool_result(
        "get_cart", {"structuredContent": NON_EMPTY_CART}
    )
    mock_instamart_tool_call.configure_tool_result(
        "checkout",
        {
            "structuredContent": {
                "orderId": f"swiggy-order-{uuid.uuid4()}",
                "total": "120.00",
            }
        },
    )


async def _add_ingredient(db_session: AsyncSession, plan: CartPlan, name: str) -> None:
    db_session.add(
        CartPlanItem(plan_id=plan.id, ingredient_name=name, match_status="unavailable")
    )
    await db_session.commit()


async def test_placing_an_order_stamps_the_matching_pantry_item(
    db_session: AsyncSession,
    linked_user: User,
    chat_session: ChatSession,
    cart_plan: CartPlan,
    placed_checkout: None,
) -> None:
    rice = await service.upsert_item(
        db_session,
        linked_user.id,
        PantryItemCreate(
            name="Basmati Rice", category=PantryCategory.GRAINS_AND_PULSES
        ),
    )
    await _add_ingredient(db_session, cart_plan, "basmati rice")

    order, _ = await place_order(
        db_session, linked_user.id, chat_session.id, cart_plan.id, "addr-1"
    )

    assert order.status == "placed"
    await db_session.refresh(rice)
    assert rice.last_bought_at == order.placed_at


async def test_an_untracked_ingredient_stamps_nothing(
    db_session: AsyncSession,
    linked_user: User,
    chat_session: ChatSession,
    cart_plan: CartPlan,
    placed_checkout: None,
) -> None:
    rice = await service.upsert_item(
        db_session,
        linked_user.id,
        PantryItemCreate(
            name="Basmati Rice", category=PantryCategory.GRAINS_AND_PULSES
        ),
    )
    await _add_ingredient(db_session, cart_plan, "Paneer")

    order, _ = await place_order(
        db_session, linked_user.id, chat_session.id, cart_plan.id, "addr-1"
    )

    assert order.status == "placed"
    await db_session.refresh(rice)
    assert rice.last_bought_at is None


async def test_a_user_with_no_pantry_still_checks_out(
    db_session: AsyncSession,
    linked_user: User,
    chat_session: ChatSession,
    cart_plan: CartPlan,
    placed_checkout: None,
) -> None:
    """The stamp is bookkeeping; an empty pantry must not affect checkout."""
    await _add_ingredient(db_session, cart_plan, "Basmati Rice")

    order, _ = await place_order(
        db_session, linked_user.id, chat_session.id, cart_plan.id, "addr-1"
    )

    assert order.status == "placed"


async def test_a_failed_order_stamps_nothing(
    db_session: AsyncSession,
    linked_user: User,
    chat_session: ChatSession,
    cart_plan: CartPlan,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    rice = await service.upsert_item(
        db_session,
        linked_user.id,
        PantryItemCreate(
            name="Basmati Rice", category=PantryCategory.GRAINS_AND_PULSES
        ),
    )
    await _add_ingredient(db_session, cart_plan, "Basmati Rice")
    mock_instamart_tool_call.configure_tool_result(
        "get_cart", {"structuredContent": NON_EMPTY_CART}
    )
    mock_instamart_tool_call.configure_tool_status("checkout", 400)

    order, _ = await place_order(
        db_session, linked_user.id, chat_session.id, cart_plan.id, "addr-1"
    )

    assert order.status != "placed"
    await db_session.refresh(rice)
    assert rice.last_bought_at is None
