"""Order history endpoints and their status/name projection (CUE-41, CUE-42)."""

from __future__ import annotations

from decimal import Decimal

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.orders import service
from app.orders.schemas import OrderStatus
from tests.conftest import InstamartToolCallStub

RAW_ORDERS = [
    {
        "orderId": "order-a",
        "status": "OUT_FOR_DELIVERY",
        "orderedTime": "2026-07-23T13:15:00+05:30",
        "grandTotal": "285.00",
        "items": [
            {"productName": "Paneer, 200g"},
            {"name": "Amul butter, 100g"},
            {"productName": "Tomatoes, 4 pcs"},
        ],
    },
    {
        "orderId": "order-b",
        "status": "DELIVERED",
        "orderedTime": "2026-07-22T11:20:00+05:30",
        "grandTotal": "134.00",
        "items": [{"productName": "Toor dal, 1kg"}],
    },
]

RAW_ORDER_DETAILS = {
    "orderId": "order-b",
    "status": "DELIVERED",
    "orderedTime": "2026-07-22T11:20:00+05:30",
    "items": [
        {"productName": "Toor dal, 1kg", "quantity": 1, "price": "110.00"},
        {"productName": "Turmeric powder, 50g", "quantity": 2, "price": "24.00"},
    ],
    "itemTotal": "158.00",
    "deliveryFee": "25.00",
    "handlingFee": "5.00",
    "grandTotal": "188.00",
}


# --- status mapping ---


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("DELIVERED", OrderStatus.DELIVERED),
        ("order_delivered", OrderStatus.DELIVERED),
        ("CANCELLED", OrderStatus.CANCELLED),
        ("canceled", OrderStatus.CANCELLED),
        ("refunded", OrderStatus.CANCELLED),
        ("OUT_FOR_DELIVERY", OrderStatus.OUT_FOR_DELIVERY),
        ("out for delivery", OrderStatus.OUT_FOR_DELIVERY),
        ("dispatched", OrderStatus.OUT_FOR_DELIVERY),
        ("PREPARING", OrderStatus.PREPARING),
        ("packing", OrderStatus.PREPARING),
        ("", OrderStatus.PREPARING),
    ],
)
def test_map_order_status(raw: str, expected: OrderStatus) -> None:
    assert service._map_order_status(raw) == expected


def test_unknown_status_falls_back_to_a_live_stage_not_a_terminal_one() -> None:
    """Guessing terminal would strand a real in-flight order on the static
    detail screen with no route into tracking - so an unrecognized status
    must land on a non-terminal value."""
    mapped = service._map_order_status("some_status_swiggy_added_later")

    assert mapped in {OrderStatus.PREPARING, OrderStatus.OUT_FOR_DELIVERY}


# --- service projection ---


async def test_list_orders_projects_status_names_and_total(
    db_session: AsyncSession,
    linked_user: User,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure(
        result={"structuredContent": {"orders": RAW_ORDERS}}
    )

    orders = await service.list_orders(db_session, linked_user.id)

    assert [o.order_id for o in orders] == ["order-a", "order-b"]
    assert orders[0].status is OrderStatus.OUT_FOR_DELIVERY
    assert orders[1].status is OrderStatus.DELIVERED
    assert orders[0].placed_at == "2026-07-23T13:15:00+05:30"
    assert orders[0].total == Decimal("285.00")
    # `name` is accepted alongside `productName` - Swiggy pins neither.
    assert orders[0].items == ["Paneer, 200g", "Amul butter, 100g", "Tomatoes, 4 pcs"]


async def test_list_orders_drops_items_carrying_no_usable_name(
    db_session: AsyncSession,
    linked_user: User,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    """A nameless entry must not become a blank segment in the list frame's
    item summary."""
    mock_instamart_tool_call.configure(
        result={
            "structuredContent": {
                "orders": [
                    {
                        "orderId": "order-c",
                        "status": "DELIVERED",
                        "items": [
                            {"productName": "Onions, 1kg"},
                            {"sku": "x-1"},
                            {"productName": "   "},
                        ],
                    }
                ]
            },
        }
    )

    orders = await service.list_orders(db_session, linked_user.id)

    assert orders[0].items == ["Onions, 1kg"]


async def test_list_orders_tolerates_an_order_missing_date_and_total(
    db_session: AsyncSession,
    linked_user: User,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    """Neither field is pinned by Swiggy's docs, so a payload without them
    still has to render a row rather than fail the whole list."""
    mock_instamart_tool_call.configure(
        result={
            "structuredContent": {
                "orders": [{"orderId": "order-d", "status": "DELIVERED"}]
            },
        }
    )

    orders = await service.list_orders(db_session, linked_user.id)

    assert orders[0].placed_at is None
    assert orders[0].total is None
    assert orders[0].items == []


# --- endpoints ---


async def test_list_orders_requires_authentication(client: httpx.AsyncClient) -> None:
    response = await client.get("/orders")

    assert response.status_code == 401


async def test_get_order_requires_authentication(client: httpx.AsyncClient) -> None:
    response = await client.get("/orders/order-b")

    assert response.status_code == 401


async def test_list_orders_returns_rows(
    authed_client: httpx.AsyncClient,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure(
        result={"structuredContent": {"orders": RAW_ORDERS}}
    )

    response = await authed_client.get("/orders")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2
    assert body[0]["order_id"] == "order-a"
    assert body[0]["status"] == "out_for_delivery"
    assert body[0]["items"][0] == "Paneer, 200g"


async def test_list_orders_returns_an_empty_list_not_an_error(
    authed_client: httpx.AsyncClient,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    """No orders is the Orders-empty frame, not a failure."""
    mock_instamart_tool_call.configure(result={"structuredContent": {"orders": []}})

    response = await authed_client.get("/orders")

    assert response.status_code == 200
    assert response.json() == []


async def test_get_order_returns_line_items_and_bill(
    authed_client: httpx.AsyncClient,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure(
        result={"structuredContent": {"order": RAW_ORDER_DETAILS}}
    )

    response = await authed_client.get("/orders/order-b")

    assert response.status_code == 200
    body = response.json()
    assert body["order_id"] == "order-b"
    assert body["status"] == "delivered"
    assert body["placed_at"] == "2026-07-22T11:20:00+05:30"
    assert body["items"] == [
        {"product_name": "Toor dal, 1kg", "quantity": 1, "price": "110.00"},
        {"product_name": "Turmeric powder, 50g", "quantity": 2, "price": "24.00"},
    ]
    assert body["item_total"] == "158.00"
    assert body["delivery_fee"] == "25.00"
    assert body["grand_total"] == "188.00"


async def test_get_order_surfaces_an_unknown_order_as_422(
    authed_client: httpx.AsyncClient,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    """Deep-linking to an order that isn't the caller's must reach the app as
    a 422 it can render an error state from, not a raw 500."""
    mock_instamart_tool_call.configure(
        result={"isError": True, "content": [{"type": "text", "text": "no such order"}]}
    )

    response = await authed_client.get("/orders/order-zzz")

    assert response.status_code == 422
