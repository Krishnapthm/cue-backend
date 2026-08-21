from __future__ import annotations

from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.cart.schemas import MatchStatus, SelectedVariant
from app.cart.service import compose_cart
from app.models.cart import CartPlan, CartPlanItem
from app.models.chat import ChatSession
from app.models.user import User
from tests.cart.conftest import FakeInstamart
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


async def test_compose_cart_subtotal_reflects_a_realistically_nested_price(
    db_session: AsyncSession,
    linked_user: User,
    chat_session: ChatSession,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    """Live `get_cart`/`update_cart` responses price a line with flat
    `mrp`/`discountedFinalPrice` siblings, not a `price` field at all (CUE-77
    live capture). A cart that doesn't resolve price from those siblings
    parses every line with `price=None`, so `subtotal` silently reads as 0
    regardless of what is actually in the cart."""
    mock_instamart_tool_call.configure_text_envelope(
        {
            "success": True,
            "data": {
                "cart": {
                    "items": [
                        {
                            "spinId": "spin-1",
                            "quantity": 2,
                            "mrp": 80,
                            "discountedFinalPrice": 60,
                        }
                    ],
                    "availablePaymentMethods": ["COD"],
                }
            },
        }
    )

    result = await compose_cart(
        db_session, linked_user.id, chat_session.id, "addr-1", [_variant()]
    )

    assert result.subtotal == Decimal("120.00")
    assert result.below_minimum is False


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

    # get_cart to read the current cart to merge onto, then update_cart with
    # the merged lines - update_cart's own response is trusted as the
    # resulting cart, so there is no third, confirming read.
    tool_names = [
        call[1]["json"]["params"]["name"] for call in mock_instamart_tool_call.calls
    ]
    assert tool_names == ["get_cart", "update_cart"]


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
    # get_cart to merge onto, update_cart raises for review, then a fallback
    # get_cart since the raised call's own response can't be trusted here.
    assert [
        call[1]["json"]["params"]["name"] for call in mock_instamart_tool_call.calls
    ] == ["get_cart", "update_cart", "get_cart"]


async def test_compose_cart_still_writes_below_the_minimum_and_reports_shortfall(
    db_session: AsyncSession,
    linked_user: User,
    chat_session: ChatSession,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    """The minimum no longer gates the write - only the reported shortfall.

    A merge can clear the minimum on the strength of what the cart already
    held even when this turn's own addition alone would not, so compose must
    always push what it has rather than guessing from its own subtotal.
    """
    below_minimum_cart = {
        "items": [{"spinId": "spin-1", "quantity": 1, "price": "50.00"}],
        "total": "50.00",
        "availablePaymentMethods": ["COD"],
    }
    mock_instamart_tool_call.configure(result={"structuredContent": below_minimum_cart})
    below_minimum_variant = _variant(unit_price="30.00", quantity=1)

    result = await compose_cart(
        db_session, linked_user.id, chat_session.id, "addr-1", [below_minimum_variant]
    )

    assert result.below_minimum is True
    assert result.subtotal == Decimal("50.00")
    assert result.shortfall == Decimal("49.00")
    assert result.cart is not None
    tool_names = [
        call[1]["json"]["params"]["name"] for call in mock_instamart_tool_call.calls
    ]
    assert tool_names == ["get_cart", "update_cart"]

    # The plan is still recorded alongside the write.
    plan = await db_session.get(CartPlan, result.plan_id)
    assert plan is not None


async def test_compose_cart_excludes_unavailable_ingredients_from_the_cart_write(
    db_session: AsyncSession,
    linked_user: User,
    chat_session: ChatSession,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    # An empty current cart, distinct from the post-write RAW_CART response:
    # otherwise the leading get_cart (read to merge onto) would itself answer
    # with spin-1 already present, and the merge would double its quantity.
    mock_instamart_tool_call.configure_tool_result(
        "get_cart",
        {"structuredContent": {"items": [], "availablePaymentMethods": ["COD"]}},
    )
    mock_instamart_tool_call.configure_tool_result(
        "update_cart", {"structuredContent": RAW_CART}
    )
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
    # calls[0] is the leading get_cart read to merge onto; calls[1] is the
    # update_cart write, which must exclude the unavailable ingredient.
    (_, update_kwargs) = mock_instamart_tool_call.calls[1]
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


async def test_compose_cart_merges_onto_items_already_in_the_cart(
    db_session: AsyncSession,
    linked_user: User,
    chat_session: ChatSession,
    fake_instamart: FakeInstamart,
) -> None:
    """A line added from the pantry screen must survive a chat recompose."""
    fake_instamart.items = {"spin-pantry": 1}

    result = await compose_cart(
        db_session, linked_user.id, chat_session.id, "addr-1", [_variant()]
    )

    assert fake_instamart.items == {"spin-pantry": 1, "spin-1": 2}
    assert result.cart is not None
    assert {line.spin_id for line in result.cart.items} == {"spin-pantry", "spin-1"}


async def test_compose_cart_never_removes_a_line_on_recompose(
    db_session: AsyncSession,
    linked_user: User,
    chat_session: ChatSession,
    fake_instamart: FakeInstamart,
) -> None:
    """A recompose that drops an ingredient must not delete its cart line.

    Removing a line is always an explicit user action (`remove_item`,
    `clear_cart`); compose only ever adds or increases one.
    """
    await compose_cart(
        db_session, linked_user.id, chat_session.id, "addr-1", [_variant()]
    )
    assert fake_instamart.items == {"spin-1": 2}

    salt = _variant("salt", spin_id="spin-2", unit_price="10.00", quantity=1)
    await compose_cart(db_session, linked_user.id, chat_session.id, "addr-1", [salt])

    assert fake_instamart.items == {"spin-1": 2, "spin-2": 1}
