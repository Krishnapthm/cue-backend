from __future__ import annotations

from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.instamart import service
from app.instamart.exceptions import (
    InstamartAuthError,
    InstamartDomainError,
    InstamartTransportError,
)
from app.models.user import User
from app.providers import service as provider_service
from tests.conftest import InstamartToolCallStub


async def test_checkout_sends_the_selected_address_id(
    db_session: AsyncSession,
    linked_user: User,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure(
        result={"success": True, "data": {"orderId": "order-1", "total": "150.00"}}
    )

    await service.checkout(db_session, linked_user.id, address_id="addr-1")

    (_, kwargs) = mock_instamart_tool_call.calls[0]
    assert kwargs["json"]["params"] == {
        "name": "checkout",
        "arguments": {"addressId": "addr-1"},
    }


async def test_checkout_parses_the_order_reference(
    db_session: AsyncSession,
    linked_user: User,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure(
        result={"success": True, "data": {"orderId": "order-1", "total": "150.00"}}
    )

    result = await service.checkout(db_session, linked_user.id, address_id="addr-1")

    assert result.order_id == "order-1"
    assert result.total == Decimal("150.00")


async def test_checkout_raises_transport_error_on_5xx_without_retrying(
    db_session: AsyncSession,
    linked_user: User,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure(status_code=503)

    with pytest.raises(InstamartTransportError):
        await service.checkout(db_session, linked_user.id, address_id="addr-1")

    assert len(mock_instamart_tool_call.calls) == 1


async def test_checkout_raises_auth_error_when_not_linked(
    db_session: AsyncSession,
    user: User,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    with pytest.raises(InstamartAuthError):
        await service.checkout(db_session, user.id, address_id="addr-1")

    assert mock_instamart_tool_call.calls == []


async def test_get_orders_sends_the_documented_count_parameter(
    db_session: AsyncSession,
    linked_user: User,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure(result={"success": True, "data": {"orders": []}})

    await service.get_orders(db_session, linked_user.id, count=5)

    (_, kwargs) = mock_instamart_tool_call.calls[0]
    assert kwargs["json"]["params"] == {
        "name": "get_orders",
        "arguments": {"count": 5, "orderType": "INSTAMART", "activeOnly": False},
    }


async def test_get_orders_parses_a_nested_list(
    db_session: AsyncSession,
    linked_user: User,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure(
        result={
            "success": True,
            "data": {
                "orders": [
                    {
                        "orderId": "order-1",
                        "status": "placed",
                        "items": [{"productName": "Milk", "quantity": 2}],
                        "address": {"addressId": "addr-1"},
                    }
                ]
            },
        }
    )

    orders = await service.get_orders(db_session, linked_user.id)

    assert len(orders) == 1
    assert orders[0].order_id == "order-1"
    assert orders[0].status == "placed"
    assert orders[0].items == [{"productName": "Milk", "quantity": 2}]


async def test_get_orders_marks_link_expired_on_live_auth_failure(
    db_session: AsyncSession,
    linked_user: User,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure(status_code=401)

    with pytest.raises(InstamartAuthError):
        await service.get_orders(db_session, linked_user.id)

    link = await provider_service.get_link(db_session, linked_user.id)
    assert link is not None
    assert link.status == "expired"


async def test_get_orders_clamps_count_above_the_documented_maximum(
    db_session: AsyncSession,
    linked_user: User,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure(result={"success": True, "data": {"orders": []}})

    await service.get_orders(db_session, linked_user.id, count=50)

    (_, kwargs) = mock_instamart_tool_call.calls[0]
    assert kwargs["json"]["params"]["arguments"]["count"] == 20


async def test_get_orders_default_call_sends_instamart_order_type(
    db_session: AsyncSession,
    linked_user: User,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure(result={"success": True, "data": {"orders": []}})

    await service.get_orders(db_session, linked_user.id)

    (_, kwargs) = mock_instamart_tool_call.calls[0]
    assert kwargs["json"]["params"]["arguments"]["orderType"] == "INSTAMART"
    assert kwargs["json"]["params"]["arguments"]["activeOnly"] is False


async def test_get_order_details_sends_the_order_id_and_parses_line_items(
    db_session: AsyncSession,
    linked_user: User,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure(
        result={
            "success": True,
            "data": {
                "order": {
                    "orderId": "order-1",
                    "status": "delivered",
                    "items": [
                        {"productName": "Milk", "quantity": 2, "price": "60.00"},
                        {"productName": "Bread", "quantity": 1, "price": "40.00"},
                    ],
                    "itemTotal": "100.00",
                    "deliveryFee": "20.00",
                    "handlingFee": "5.00",
                    "grandTotal": "125.00",
                }
            },
        }
    )

    details = await service.get_order_details(db_session, linked_user.id, "order-1")

    (_, kwargs) = mock_instamart_tool_call.calls[0]
    assert kwargs["json"]["params"] == {
        "name": "get_order_details",
        "arguments": {"orderId": "order-1"},
    }
    assert details.order_id == "order-1"
    assert details.status == "delivered"
    assert len(details.items) == 2
    assert details.items[0].product_name == "Milk"
    assert details.items[0].quantity == 2
    assert details.items[0].price == Decimal("60.00")
    assert details.item_total == Decimal("100.00")
    assert details.grand_total == Decimal("125.00")


async def test_get_order_details_raises_domain_error_on_success_false(
    db_session: AsyncSession,
    linked_user: User,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure(
        result={"success": False, "error": {"message": "Order not found"}}
    )

    with pytest.raises(InstamartDomainError):
        await service.get_order_details(db_session, linked_user.id, "order-999")


async def test_get_order_details_raises_auth_error_when_not_linked(
    db_session: AsyncSession,
    user: User,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    with pytest.raises(InstamartAuthError):
        await service.get_order_details(db_session, user.id, "order-1")

    assert mock_instamart_tool_call.calls == []
