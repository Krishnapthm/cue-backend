from __future__ import annotations

from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.cart.schemas import MatchStatus, SelectedVariant
from app.cart.service import compose_cart
from app.models.cart import CartPlan, CartPlanItem
from app.models.chat import ChatSession
from app.models.user import User
from tests.conftest import InstamartToolCallStub

RAW_CART = {
    "items": [{"spinId": "spin-1", "quantity": 2, "price": "60.00"}],
    "total": "120.00",
    "minimumOrderValue": "99.00",
    "availablePaymentMethods": ["COD"],
}


def _variant(
    name: str = "sugar",
    *,
    spin_id: str | None = "spin-1",
    unit_price: str | None = "60.00",
    quantity: int | None = 2,
    match_status: MatchStatus = MatchStatus.MATCHED,
) -> SelectedVariant:
    return SelectedVariant(
        ingredient_name=name,
        match_status=match_status,
        spin_id=spin_id,
        product_name="Product",
        pack_size="500 g",
        unit_price=Decimal(unit_price) if unit_price is not None else None,
        quantity=quantity,
        selection_reason="test selection",
    )


async def test_compose_cart_persists_the_plan_and_its_items(
    db_session: AsyncSession,
    linked_user: User,
    chat_session: ChatSession,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure(result={"structuredContent": RAW_CART})

    result = await compose_cart(
        db_session, linked_user.id, chat_session.id, "addr-1", [_variant()]
    )

    plan = await db_session.get(CartPlan, result.plan_id)
    assert plan is not None
    assert plan.address_id == "addr-1"
    assert plan.superseded_at is None

    items = (
        (
            await db_session.execute(
                select(CartPlanItem).where(CartPlanItem.plan_id == plan.id)
            )
        )
        .scalars()
        .all()
    )
    assert len(items) == 1
    assert items[0].spin_id == "spin-1"
    assert items[0].match_status == "matched"


async def test_compose_cart_writes_the_full_cart_and_reads_it_back(
    db_session: AsyncSession,
    linked_user: User,
    chat_session: ChatSession,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure(result={"structuredContent": RAW_CART})

    result = await compose_cart(
        db_session, linked_user.id, chat_session.id, "addr-1", [_variant()]
    )

    assert result.below_minimum is False
    assert result.subtotal == Decimal("120.00")
    assert result.cart is not None
    assert result.cart.total == Decimal("120.00")

    # update_cart, then get_cart - both called, update_cart first.
    tool_names = [
        call[1]["json"]["params"]["name"] for call in mock_instamart_tool_call.calls
    ]
    assert tool_names == ["update_cart", "get_cart"]


async def test_compose_cart_reads_back_an_adjusted_cart_for_user_review(
    db_session: AsyncSession,
    linked_user: User,
    chat_session: ChatSession,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure_text_envelope(
        {
            "success": False,
            "error": {
                "message": (
                    "Some items have limited stock and their quantities were "
                    "adjusted. Please review your cart. Cart updated successfully."
                )
            },
        },
        tool_name="update_cart",
    )
    mock_instamart_tool_call.configure_tool_result(
        "get_cart", {"structuredContent": RAW_CART}
    )

    result = await compose_cart(
        db_session, linked_user.id, chat_session.id, "addr-1", [_variant()]
    )

    assert result.cart is not None
    assert result.cart.total == Decimal("120.00")
    assert [
        call[1]["json"]["params"]["name"] for call in mock_instamart_tool_call.calls
    ] == ["update_cart", "get_cart"]


async def test_compose_cart_never_calls_update_cart_below_the_minimum(
    db_session: AsyncSession,
    linked_user: User,
    chat_session: ChatSession,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    below_minimum_variant = _variant(unit_price="30.00", quantity=1)

    result = await compose_cart(
        db_session, linked_user.id, chat_session.id, "addr-1", [below_minimum_variant]
    )

    assert result.below_minimum is True
    assert result.subtotal == Decimal("30.00")
    assert result.shortfall == Decimal("69.00")
    assert result.cart is None
    assert mock_instamart_tool_call.calls == []

    # The plan is still recorded even though nothing was written to Swiggy.
    plan = await db_session.get(CartPlan, result.plan_id)
    assert plan is not None


async def test_compose_cart_excludes_unavailable_ingredients_from_the_cart_write(
    db_session: AsyncSession,
    linked_user: User,
    chat_session: ChatSession,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure(result={"structuredContent": RAW_CART})
    unavailable = _variant(
        "cardamom",
        spin_id=None,
        unit_price=None,
        quantity=None,
        match_status=MatchStatus.UNAVAILABLE,
    )

    result = await compose_cart(
        db_session,
        linked_user.id,
        chat_session.id,
        "addr-1",
        [_variant(), unavailable],
    )

    assert result.subtotal == Decimal("120.00")
    (_, update_kwargs) = mock_instamart_tool_call.calls[0]
    assert update_kwargs["json"]["params"]["arguments"]["items"] == [
        {"spinId": "spin-1", "quantity": 2}
    ]


async def test_compose_cart_supersedes_the_previous_live_plan_on_recompose(
    db_session: AsyncSession,
    linked_user: User,
    chat_session: ChatSession,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure(result={"structuredContent": RAW_CART})

    first = await compose_cart(
        db_session, linked_user.id, chat_session.id, "addr-1", [_variant()]
    )
    second = await compose_cart(
        db_session, linked_user.id, chat_session.id, "addr-2", [_variant()]
    )

    first_plan = await db_session.get(CartPlan, first.plan_id)
    second_plan = await db_session.get(CartPlan, second.plan_id)
    assert first_plan is not None
    assert second_plan is not None
    assert first_plan.superseded_at is not None
    assert second_plan.superseded_at is None
    assert second_plan.address_id == "addr-2"

    live_plans = (
        (
            await db_session.execute(
                select(CartPlan).where(
                    CartPlan.session_id == chat_session.id,
                    CartPlan.superseded_at.is_(None),
                )
            )
        )
        .scalars()
        .all()
    )
    assert len(live_plans) == 1
