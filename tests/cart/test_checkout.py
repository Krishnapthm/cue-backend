from __future__ import annotations

from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.cart.checkout import place_order
from app.cart.exceptions import CartNotCheckoutableError, CheckoutInProgressError
from app.instamart.exceptions import InstamartAuthError
from app.models.cart import CartPlan
from app.models.chat import ChatSession
from app.models.order import Order
from app.models.user import User
from tests.conftest import InstamartToolCallStub

NON_EMPTY_CART = {
    "items": [{"spinId": "spin-1", "quantity": 2, "price": "60.00"}],
    "total": "120.00",
    "availablePaymentMethods": ["COD"],
}
EMPTY_CART = {"items": [], "availablePaymentMethods": ["COD"]}


async def test_place_order_places_a_cod_order_on_success(
    db_session: AsyncSession,
    linked_user: User,
    chat_session: ChatSession,
    cart_plan: CartPlan,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure_tool_result(
        "get_cart", {"structuredContent": NON_EMPTY_CART}
    )
    mock_instamart_tool_call.configure_tool_result(
        "checkout",
        {
            "structuredContent": {"orderId": "swiggy-order-1", "total": "120.00"},
        },
    )

    order, recent_orders = await place_order(
        db_session, linked_user.id, chat_session.id, cart_plan.id, "addr-1"
    )

    assert order.status == "placed"
    assert order.swiggy_order_id == "swiggy-order-1"
    assert order.total == Decimal("120.00")
    assert order.placed_at is not None
    assert recent_orders == []


async def test_place_order_raises_when_the_server_cart_is_empty(
    db_session: AsyncSession,
    linked_user: User,
    chat_session: ChatSession,
    cart_plan: CartPlan,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure_tool_result(
        "get_cart", {"structuredContent": EMPTY_CART}
    )

    with pytest.raises(CartNotCheckoutableError):
        await place_order(
            db_session, linked_user.id, chat_session.id, cart_plan.id, "addr-1"
        )

    # checkout must never be called against an empty cart.
    tool_names = [
        call[1]["json"]["params"]["name"] for call in mock_instamart_tool_call.calls
    ]
    assert "checkout" not in tool_names


async def test_place_order_raises_when_another_checkout_is_already_in_progress(
    db_session: AsyncSession,
    linked_user: User,
    chat_session: ChatSession,
    cart_plan: CartPlan,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure_tool_result(
        "get_cart", {"structuredContent": NON_EMPTY_CART}
    )
    db_session.add(
        Order(
            user_id=linked_user.id,
            session_id=chat_session.id,
            plan_id=cart_plan.id,
            address_id="addr-1",
            status="placing",
        )
    )
    await db_session.commit()

    with pytest.raises(CheckoutInProgressError):
        await place_order(
            db_session, linked_user.id, chat_session.id, cart_plan.id, "addr-1"
        )


async def test_place_order_marks_unknown_and_checks_get_orders_on_transport_failure(
    db_session: AsyncSession,
    linked_user: User,
    chat_session: ChatSession,
    cart_plan: CartPlan,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure_tool_result(
        "get_cart", {"structuredContent": NON_EMPTY_CART}
    )
    mock_instamart_tool_call.configure_tool_status("checkout", 503)
    mock_instamart_tool_call.configure_tool_result(
        "get_orders",
        {
            "structuredContent": {
                "orders": [{"orderId": "swiggy-order-1", "status": "placed"}]
            },
        },
    )

    order, recent_orders = await place_order(
        db_session, linked_user.id, chat_session.id, cart_plan.id, "addr-1"
    )

    assert order.status == "unknown"
    assert order.swiggy_order_id is None
    assert len(recent_orders) == 1
    assert recent_orders[0].order_id == "swiggy-order-1"

    # checkout is called exactly once - never retried within place_order.
    checkout_calls = [
        call
        for call in mock_instamart_tool_call.calls
        if call[1]["json"]["params"]["name"] == "checkout"
    ]
    assert len(checkout_calls) == 1


async def test_place_order_marks_failed_on_a_domain_error(
    db_session: AsyncSession,
    linked_user: User,
    chat_session: ChatSession,
    cart_plan: CartPlan,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure_tool_result(
        "get_cart", {"structuredContent": NON_EMPTY_CART}
    )
    mock_instamart_tool_call.configure_tool_result(
        "checkout",
        {"isError": True, "content": [{"type": "text", "text": "Store closed"}]},
    )

    order, recent_orders = await place_order(
        db_session, linked_user.id, chat_session.id, cart_plan.id, "addr-1"
    )

    assert order.status == "failed"
    assert recent_orders == []


async def test_place_order_marks_unknown_when_checkout_succeeds_without_an_order_id(
    db_session: AsyncSession,
    linked_user: User,
    chat_session: ChatSession,
    cart_plan: CartPlan,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure_tool_result(
        "get_cart", {"structuredContent": NON_EMPTY_CART}
    )
    mock_instamart_tool_call.configure_tool_result(
        "checkout", {"structuredContent": {}}
    )
    mock_instamart_tool_call.configure_tool_result(
        "get_orders", {"structuredContent": {"orders": []}}
    )

    order, recent_orders = await place_order(
        db_session, linked_user.id, chat_session.id, cart_plan.id, "addr-1"
    )

    assert order.status == "unknown"
    assert recent_orders == []


async def test_place_order_raises_auth_error_when_not_linked(
    db_session: AsyncSession,
    user: User,
    chat_session: ChatSession,
    cart_plan: CartPlan,
) -> None:
    with pytest.raises(InstamartAuthError):
        await place_order(db_session, user.id, chat_session.id, cart_plan.id, "addr-1")
