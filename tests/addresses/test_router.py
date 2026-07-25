"""HTTP surface of the address endpoints (CUE-66).

The wrappers' own behavior - response-envelope handling, camelCase aliasing
of the create payload - is covered in `tests/instamart/test_service.py`;
these tests cover only what the router adds: auth, status codes, and the
shape the app actually receives.
"""

from __future__ import annotations

import httpx

from tests.conftest import InstamartToolCallStub

RAW_ADDRESS = {
    "addressId": "addr-1",
    "fullAddress": "221B Baker Street, London",
    "addressLine": "221B Baker Street",
    "city": "London",
    "postalCode": "NW1 6XE",
    "addressCategory": "HOME",
}

CREATE_BODY = {
    "fullAddress": "221B Baker Street, London",
    "addressLine": "221B Baker Street",
    "city": "London",
    "postalCode": "NW1 6XE",
    "latitude": 51.5237,
    "longitude": -0.1585,
    "addressCategory": "HOME",
    "userName": "Sherlock",
    "userPhone": "9000000000",
}


async def test_list_requires_authentication(client: httpx.AsyncClient) -> None:
    response = await client.get("/addresses")

    assert response.status_code == 401


async def test_list_returns_saved_addresses(
    authed_client: httpx.AsyncClient,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure(
        result={"structuredContent": {"addresses": [RAW_ADDRESS]}}
    )

    response = await authed_client.get("/addresses")

    assert response.status_code == 200
    assert response.json() == [
        {
            "addressId": "addr-1",
            "fullAddress": "221B Baker Street, London",
            "addressLine": "221B Baker Street",
            "addressLine2": None,
            "city": "London",
            "postalCode": "NW1 6XE",
            "locality": None,
            "addressCategory": "HOME",
            "addressTag": None,
        }
    ]


async def test_list_returns_empty_list_when_no_addresses_saved(
    authed_client: httpx.AsyncClient,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    """An account with no saved addresses is a normal state, not an error -
    it is what the empty address sheet renders."""
    mock_instamart_tool_call.configure(result={"structuredContent": {"addresses": []}})

    response = await authed_client.get("/addresses")

    assert response.status_code == 200
    assert response.json() == []


async def test_create_requires_authentication(client: httpx.AsyncClient) -> None:
    response = await client.post("/addresses", json=CREATE_BODY)

    assert response.status_code == 401


async def test_create_returns_the_saved_address(
    authed_client: httpx.AsyncClient,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure(
        result={"structuredContent": {"address": RAW_ADDRESS}}
    )

    response = await authed_client.post("/addresses", json=CREATE_BODY)

    assert response.status_code == 201
    assert response.json()["addressId"] == "addr-1"

    (_, kwargs) = mock_instamart_tool_call.calls[0]
    arguments = kwargs["json"]["params"]["arguments"]
    assert arguments["fullAddress"] == "221B Baker Street, London"
    assert arguments["userPhone"] == "9000000000"


async def test_create_rejects_an_incomplete_body(
    authed_client: httpx.AsyncClient,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    """A missing required field is caught by validation, before Swiggy."""
    body = {key: value for key, value in CREATE_BODY.items() if key != "postalCode"}

    response = await authed_client.post("/addresses", json=body)

    assert response.status_code == 422
    assert mock_instamart_tool_call.calls == []


async def test_create_surfaces_swiggy_domain_failure_as_422(
    authed_client: httpx.AsyncClient,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure(
        result={
            "isError": True,
            "content": [{"type": "text", "text": "unserviceable pincode"}],
        }
    )

    response = await authed_client.post("/addresses", json=CREATE_BODY)

    assert response.status_code == 422


async def test_delete_requires_authentication(client: httpx.AsyncClient) -> None:
    response = await client.delete("/addresses/addr-1")

    assert response.status_code == 401


async def test_delete_returns_no_content(
    authed_client: httpx.AsyncClient,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure(result={"structuredContent": {}})

    response = await authed_client.delete("/addresses/addr-1")

    assert response.status_code == 204
    assert response.content == b""

    (_, kwargs) = mock_instamart_tool_call.calls[0]
    assert kwargs["json"]["params"] == {
        "name": "delete_address",
        "arguments": {"addressId": "addr-1"},
    }


async def test_delete_surfaces_unknown_address_as_422(
    authed_client: httpx.AsyncClient,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    """An unknown id must surface Swiggy's own refusal, not a 500."""
    mock_instamart_tool_call.configure(
        result={
            "isError": True,
            "content": [{"type": "text", "text": "no such address"}],
        }
    )

    response = await authed_client.delete("/addresses/does-not-exist")

    assert response.status_code == 422
