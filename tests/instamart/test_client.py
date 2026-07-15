from __future__ import annotations

import httpx
import pytest

from app.instamart import client
from app.instamart.exceptions import (
    InstamartAuthError,
    InstamartDomainError,
    InstamartTransportError,
)
from tests.conftest import InstamartToolCallStub


async def test_call_tool_returns_the_data_payload_on_success(
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure(
        result={"success": True, "data": {"addresses": []}}
    )

    data = await client.call_tool("at_token", "get_addresses", {})

    assert data == {"addresses": []}


async def test_call_tool_sends_a_bearer_token_and_jsonrpc_body(
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    await client.call_tool("at_token", "get_addresses", {"foo": "bar"})

    (_, kwargs) = mock_instamart_tool_call.calls[0]
    assert kwargs["headers"]["Authorization"] == "Bearer at_token"
    assert kwargs["json"]["method"] == "tools/call"
    assert kwargs["json"]["params"] == {
        "name": "get_addresses",
        "arguments": {"foo": "bar"},
    }


@pytest.mark.parametrize("status_code", [401, 419])
async def test_call_tool_raises_auth_error_on_recoverable_http_status(
    mock_instamart_tool_call: InstamartToolCallStub, status_code: int
) -> None:
    mock_instamart_tool_call.configure(status_code=status_code)

    with pytest.raises(InstamartAuthError):
        await client.call_tool("at_token", "get_addresses", {})


async def test_call_tool_raises_auth_error_on_jsonrpc_auth_error_code(
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure(rpc_error={"code": -32001, "message": "auth"})

    with pytest.raises(InstamartAuthError):
        await client.call_tool("at_token", "get_addresses", {})


@pytest.mark.parametrize("status_code", [500, 502, 503, 504])
async def test_call_tool_raises_transport_error_on_upstream_5xx(
    mock_instamart_tool_call: InstamartToolCallStub, status_code: int
) -> None:
    mock_instamart_tool_call.configure(status_code=status_code)

    with pytest.raises(InstamartTransportError):
        await client.call_tool("at_token", "get_addresses", {})


async def test_call_tool_raises_transport_error_on_network_failure(
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure(raises=httpx.ConnectTimeout("timed out"))

    with pytest.raises(InstamartTransportError):
        await client.call_tool("at_token", "get_addresses", {})


async def test_call_tool_raises_transport_error_on_other_jsonrpc_error(
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure(
        rpc_error={"code": -32603, "message": "internal error"}
    )

    with pytest.raises(InstamartTransportError):
        await client.call_tool("at_token", "get_addresses", {})


async def test_call_tool_raises_domain_error_on_success_false(
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure(
        result={"success": False, "error": {"message": "Item out of stock"}}
    )

    with pytest.raises(InstamartDomainError, match="Item out of stock"):
        await client.call_tool("at_token", "search_products", {})
