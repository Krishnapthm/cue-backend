from __future__ import annotations

from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.instamart import service
from app.instamart.exceptions import InstamartAuthError
from app.instamart.schemas import CartItemInput
from app.models.user import User
from app.providers import service as provider_service
from tests.conftest import InstamartToolCallStub

RAW_CART = {
    "items": [
        {"spinId": "spin-1", "quantity": 2, "price": "27.00", "productName": "Milk"},
    ],
    "total": "54.00",
    "minimumOrderValue": "99.00",
    "availablePaymentMethods": ["COD"],
}


async def test_update_cart_sends_the_full_item_list_and_address(
    db_session: AsyncSession,
    linked_user: User,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure(result={"structuredContent": RAW_CART})

    await service.update_cart(
        db_session,
        linked_user.id,
        address_id="addr-1",
        items=[CartItemInput(spin_id="spin-1", quantity=2)],
    )

    (_, kwargs) = mock_instamart_tool_call.calls[0]
    assert kwargs["json"]["params"] == {
        "name": "update_cart",
        "arguments": {
            "selectedAddressId": "addr-1",
            "items": [{"spinId": "spin-1", "quantity": 2}],
        },
    }


async def test_update_cart_parses_the_returned_cart(
    db_session: AsyncSession,
    linked_user: User,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure(result={"structuredContent": RAW_CART})

    cart = await service.update_cart(
        db_session,
        linked_user.id,
        address_id="addr-1",
        items=[CartItemInput(spin_id="spin-1", quantity=2)],
    )

    assert cart.total == Decimal("54.00")
    assert cart.minimum_order_value == Decimal("99.00")
    assert cart.available_payment_methods == ["COD"]
    assert cart.items[0].spin_id == "spin-1"


async def test_get_cart_reads_back_the_server_cart(
    db_session: AsyncSession,
    linked_user: User,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure(result={"structuredContent": RAW_CART})

    cart = await service.get_cart(db_session, linked_user.id)

    (_, kwargs) = mock_instamart_tool_call.calls[0]
    assert kwargs["json"]["params"]["name"] == "get_cart"
    assert cart.total == Decimal("54.00")


async def test_update_cart_raises_auth_error_when_not_linked(
    db_session: AsyncSession,
    user: User,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    with pytest.raises(InstamartAuthError):
        await service.update_cart(db_session, user.id, address_id="addr-1", items=[])

    assert mock_instamart_tool_call.calls == []


async def test_get_cart_marks_link_expired_on_live_auth_failure(
    db_session: AsyncSession,
    linked_user: User,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure(status_code=401)

    with pytest.raises(InstamartAuthError):
        await service.get_cart(db_session, linked_user.id)

    link = await provider_service.get_link(db_session, linked_user.id)
    assert link is not None
    assert link.status == "expired"
