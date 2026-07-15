"""Swiggy Instamart MCP JSON-RPC tool client (CUE-10).

Tool calls are JSON-RPC 2.0 `tools/call` requests POSTed to the Instamart MCP
endpoint, authenticated with the caller's Swiggy access token (per
https://mcp.swiggy.com/builders/docs/start/authenticate.md). This module only
speaks JSON-RPC/HTTP; it does not know about Cue users or provider links -
callers pass in an already-resolved access token and are responsible for
routing `InstamartAuthError` through the recovery ladder (R2.5).
"""

from __future__ import annotations

import itertools
import logging
from typing import Any

import httpx

from app.instamart.constants import (
    AUTH_FAILURE_STATUS_CODES,
    INSTAMART_MCP_ENDPOINT,
    JSONRPC_AUTH_ERROR_CODE,
    REQUEST_TIMEOUT_SECONDS,
)
from app.instamart.exceptions import (
    InstamartAuthError,
    InstamartDomainError,
    InstamartTransportError,
)

logger = logging.getLogger(__name__)

# Monotonic per-process JSON-RPC request ids; Swiggy does not require
# global uniqueness, only uniqueness within a single request/response pair.
_request_ids = itertools.count(1)


async def call_tool(
    access_token: str, tool_name: str, arguments: dict[str, Any]
) -> Any:
    """Invoke one Instamart MCP tool and return its `data` payload.

    Args:
        access_token: The caller's live Swiggy access token.
        tool_name: The MCP tool to invoke (e.g. "get_addresses").
        arguments: The tool's JSON-RPC arguments.

    Returns:
        The tool result's `data` payload (shape is tool-specific).

    Raises:
        InstamartAuthError: Swiggy rejected the token (HTTP 401/419, or
            JSON-RPC error -32001).
        InstamartTransportError: The request failed at the network, HTTP, or
            JSON-RPC protocol level before a tool-level result was reached.
        InstamartDomainError: The tool ran but reported `success: false`
            (e.g. out of stock); terminal, not retryable.
    """
    body = {
        "jsonrpc": "2.0",
        "id": next(_request_ids),
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as http_client:
            response = await http_client.post(
                INSTAMART_MCP_ENDPOINT,
                json=body,
                headers={"Authorization": f"Bearer {access_token}"},
            )
    except httpx.HTTPError as exc:
        logger.warning("Instamart tool %s request failed: %s", tool_name, exc)
        raise InstamartTransportError from exc

    if response.status_code in AUTH_FAILURE_STATUS_CODES:
        raise InstamartAuthError
    if response.status_code != httpx.codes.OK:
        logger.warning(
            "Instamart tool %s returned HTTP %s", tool_name, response.status_code
        )
        raise InstamartTransportError

    payload = response.json()
    rpc_error = payload.get("error")
    if rpc_error is not None:
        if rpc_error.get("code") == JSONRPC_AUTH_ERROR_CODE:
            raise InstamartAuthError
        logger.warning("Instamart tool %s JSON-RPC error: %s", tool_name, rpc_error)
        raise InstamartTransportError(rpc_error.get("message"))

    result = payload.get("result") or {}
    if result.get("success") is False:
        error = result.get("error") or {}
        raise InstamartDomainError(error.get("message"))

    return result.get("data")
