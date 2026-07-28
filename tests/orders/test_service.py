from __future__ import annotations

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.instamart.exceptions import (
    InstamartAuthError,
    InstamartDomainError,
    InstamartTransportError,
)
from app.models.user import User
from app.orders import service
from app.orders.schemas import TrackingStatus
from tests.conftest import InstamartToolCallStub

RAW_TRACKING = {
    "orderId": "order-1",
    "status": "out_for_delivery",
    "eta": "12 mins",
    "deliveryPartnerLocation": {"lat": 12.9, "lng": 77.6},
}


def _set_clock(monkeypatch: pytest.MonkeyPatch, value: list[float]) -> None:
    """Monkeypatch `app.orders.service._now` to return `value[0]`, letting a
    test advance monotonic time by mutating the shared list - no sleeping."""
    monkeypatch.setattr(service, "_now", lambda: value[0])


async def test_first_call_hits_swiggy_and_returns_mapped_tracking(
    db_session: AsyncSession,
    linked_user: User,
    mock_instamart_tool_call: InstamartToolCallStub,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_clock(monkeypatch, [0.0])
    mock_instamart_tool_call.configure(result={"structuredContent": RAW_TRACKING})

    response = await service.get_tracking(
        db_session, linked_user.id, "order-1", lat=1.0, lng=2.0
    )

    assert len(mock_instamart_tool_call.calls) == 1
    assert response.order_id == "order-1"
    assert response.status is TrackingStatus.ACTIVE
    assert response.eta == "12 mins"
    assert response.delivery_partner_location == {"lat": 12.9, "lng": 77.6}


async def test_second_call_within_window_is_served_from_cache(
    db_session: AsyncSession,
    linked_user: User,
    mock_instamart_tool_call: InstamartToolCallStub,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]
    _set_clock(monkeypatch, clock)
    mock_instamart_tool_call.configure(result={"structuredContent": RAW_TRACKING})

    first = await service.get_tracking(
        db_session, linked_user.id, "order-1", lat=1.0, lng=2.0
    )
    clock[0] = 5.0
    second = await service.get_tracking(
        db_session, linked_user.id, "order-1", lat=9.0, lng=9.0
    )

    assert len(mock_instamart_tool_call.calls) == 1
    assert second == first


async def test_call_after_the_floor_hits_swiggy_again(
    db_session: AsyncSession,
    linked_user: User,
    mock_instamart_tool_call: InstamartToolCallStub,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]
    _set_clock(monkeypatch, clock)
    mock_instamart_tool_call.configure(result={"structuredContent": RAW_TRACKING})

    await service.get_tracking(db_session, linked_user.id, "order-1", lat=1.0, lng=2.0)
    clock[0] = 10.0
    await service.get_tracking(db_session, linked_user.id, "order-1", lat=1.0, lng=2.0)

    assert len(mock_instamart_tool_call.calls) == 2


@pytest.mark.parametrize(
    ("raw_status", "expected"),
    [
        ("delivered", TrackingStatus.DELIVERED),
        ("Completed", TrackingStatus.DELIVERED),
        ("out_for_delivery", TrackingStatus.ACTIVE),
        ("preparing", TrackingStatus.ACTIVE),
    ],
)
async def test_status_mapping(
    db_session: AsyncSession,
    linked_user: User,
    mock_instamart_tool_call: InstamartToolCallStub,
    monkeypatch: pytest.MonkeyPatch,
    raw_status: str,
    expected: TrackingStatus,
) -> None:
    _set_clock(monkeypatch, [0.0])
    mock_instamart_tool_call.configure(
        result={"structuredContent": {**RAW_TRACKING, "status": raw_status}}
    )

    response = await service.get_tracking(
        db_session, linked_user.id, "order-1", lat=1.0, lng=2.0
    )

    assert response.status is expected


async def test_a_different_order_key_is_not_served_from_another_orders_cache(
    db_session: AsyncSession,
    linked_user: User,
    mock_instamart_tool_call: InstamartToolCallStub,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_clock(monkeypatch, [0.0])
    mock_instamart_tool_call.configure(result={"structuredContent": RAW_TRACKING})

    await service.get_tracking(db_session, linked_user.id, "order-1", lat=1.0, lng=2.0)
    await service.get_tracking(
        db_session,
        linked_user.id,
        "order-2",
        lat=1.0,
        lng=2.0,
    )

    assert len(mock_instamart_tool_call.calls) == 2


async def test_instamart_domain_error_propagates_and_is_not_cached(
    db_session: AsyncSession,
    linked_user: User,
    mock_instamart_tool_call: InstamartToolCallStub,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]
    _set_clock(monkeypatch, clock)
    mock_instamart_tool_call.configure(
        result={
            "isError": True,
            "content": [{"type": "text", "text": "order too old to track"}],
        }
    )

    with pytest.raises(InstamartDomainError):
        await service.get_tracking(
            db_session, linked_user.id, "order-1", lat=1.0, lng=2.0
        )

    assert len(mock_instamart_tool_call.calls) == 1

    # A subsequent call within what would have been the poll window still
    # hits Swiggy live, because the failed call was never cached.
    clock[0] = 1.0
    with pytest.raises(InstamartDomainError):
        await service.get_tracking(
            db_session, linked_user.id, "order-1", lat=1.0, lng=2.0
        )

    assert len(mock_instamart_tool_call.calls) == 2


# --- the list floor (CUE-88) ------------------------------------------------
#
# A chat "where is my order" turn is a poll the user can trigger by typing, so
# `list_orders_throttled` puts the order list behind the same floor tracking
# uses. Same constant, same clock - the floor is defined once.

RAW_ORDERS = [
    {
        "orderId": "order-a",
        "status": "OUT_FOR_DELIVERY",
        "orderedTime": "2026-07-28T13:15:00+05:30",
        "grandTotal": "285.00",
        "items": [{"productName": "Paneer, 200g"}],
    }
]


async def test_repeated_order_status_turns_make_one_upstream_call(
    db_session: AsyncSession,
    linked_user: User,
    mock_instamart_tool_call: InstamartToolCallStub,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]
    _set_clock(monkeypatch, clock)
    mock_instamart_tool_call.configure(
        result={"structuredContent": {"orders": RAW_ORDERS}}
    )

    first = await service.list_orders_throttled(db_session, linked_user.id)
    clock[0] = 3.0
    second = await service.list_orders_throttled(db_session, linked_user.id)
    clock[0] = 9.9
    third = await service.list_orders_throttled(db_session, linked_user.id)

    # Three questions in a row, one live call: the chat path cannot become a way
    # to bypass the floor.
    assert len(mock_instamart_tool_call.calls) == 1
    assert second == first
    assert third == first


async def test_the_list_goes_live_again_once_the_floor_has_passed(
    db_session: AsyncSession,
    linked_user: User,
    mock_instamart_tool_call: InstamartToolCallStub,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]
    _set_clock(monkeypatch, clock)
    mock_instamart_tool_call.configure(
        result={"structuredContent": {"orders": RAW_ORDERS}}
    )

    await service.list_orders_throttled(db_session, linked_user.id)
    clock[0] = 10.0
    await service.list_orders_throttled(db_session, linked_user.id)

    assert len(mock_instamart_tool_call.calls) == 2


async def test_the_list_floor_is_per_user(
    db_session: AsyncSession,
    linked_user: User,
    mock_instamart_tool_call: InstamartToolCallStub,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # One user's question must never be answered with another user's orders.
    _set_clock(monkeypatch, [0.0])
    mock_instamart_tool_call.configure(
        result={"structuredContent": {"orders": RAW_ORDERS}}
    )
    other = User(firebase_uid=f"firebase-uid-{uuid.uuid4()}", email="other@example.com")
    db_session.add(other)
    await db_session.commit()

    await service.list_orders_throttled(db_session, linked_user.id)
    with pytest.raises(InstamartAuthError):
        # `other` has no Swiggy link, so a cache hit is the only way this could
        # return rows at all.
        await service.list_orders_throttled(db_session, other.id)

    assert len(mock_instamart_tool_call.calls) == 1


async def test_a_failed_list_is_never_cached(
    db_session: AsyncSession,
    linked_user: User,
    mock_instamart_tool_call: InstamartToolCallStub,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A transient failure must be retried on the next turn, not served back for
    # ten seconds.
    _set_clock(monkeypatch, [0.0])
    mock_instamart_tool_call.configure(rpc_error={"code": -32000, "message": "boom"})

    with pytest.raises(InstamartTransportError):
        await service.list_orders_throttled(db_session, linked_user.id)

    mock_instamart_tool_call.configure(
        result={"structuredContent": {"orders": RAW_ORDERS}}
    )
    orders = await service.list_orders_throttled(db_session, linked_user.id)

    assert [order.order_id for order in orders] == ["order-a"]
    assert len(mock_instamart_tool_call.calls) == 2
