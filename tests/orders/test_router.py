from __future__ import annotations

import httpx

from tests.conftest import InstamartToolCallStub

RAW_TRACKING = {
    "orderId": "order-1",
    "status": "out_for_delivery",
    "eta": "12 mins",
    "deliveryPartnerLocation": {"lat": 12.9, "lng": 77.6},
}


async def test_track_requires_authentication(client: httpx.AsyncClient) -> None:
    response = await client.get("/orders/o1/track", params={"lat": 1.0, "lng": 2.0})

    assert response.status_code == 401


async def test_track_returns_status_eta_and_delivery_location(
    authed_client: httpx.AsyncClient,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure(result={"success": True, "data": RAW_TRACKING})

    response = await authed_client.get(
        "/orders/order-1/track", params={"lat": 1.0, "lng": 2.0}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["order_id"] == "order-1"
    assert body["status"] in {"active", "delivered"}
    assert body["eta"] == "12 mins"
    assert body["delivery_partner_location"] == {"lat": 12.9, "lng": 77.6}


async def test_track_surfaces_swiggy_domain_failure_as_422(
    authed_client: httpx.AsyncClient,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure(
        result={"success": False, "error": {"message": "not this user's order"}}
    )

    response = await authed_client.get(
        "/orders/order-1/track", params={"lat": 1.0, "lng": 2.0}
    )

    assert response.status_code == 422


async def test_track_requires_lat_and_lng_query_params(
    authed_client: httpx.AsyncClient,
) -> None:
    response = await authed_client.get("/orders/order-1/track")

    assert response.status_code == 422
