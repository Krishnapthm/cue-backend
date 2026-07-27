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
import json
import logging
from typing import Any

import httpx

from app.instamart.constants import (
    AUTH_FAILURE_STATUS_CODES,
    INSTAMART_MCP_ENDPOINT,
    JSONRPC_AUTH_ERROR_CODE,
    MCP_ACCEPT_HEADER,
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


def _decode_envelope(response: httpx.Response) -> dict[str, Any]:
    """Decode a JSON-RPC envelope from a JSON or SSE response body.

    We advertise both content types (`MCP_ACCEPT_HEADER`), so the server is
    free to answer with either. An SSE body carries the envelope in the
    `data:` field of its final event; anything else is parsed as plain JSON.

    Raises:
        InstamartTransportError: The body was not decodable as a JSON-RPC
            envelope - a protocol-level failure, not a tool result.
    """
    content_type = response.headers.get("content-type", "")
    try:
        if content_type.startswith("text/event-stream"):
            payloads = [
                line.removeprefix("data:").strip()
                for line in response.text.splitlines()
                if line.startswith("data:")
            ]
            if not payloads:
                raise ValueError("SSE response carried no data event")
            envelope = json.loads(payloads[-1])
        else:
            envelope = response.json()
    except ValueError as exc:
        logger.warning("Instamart response was not valid JSON-RPC: %s", exc)
        raise InstamartTransportError from exc

    if not isinstance(envelope, dict):
        raise InstamartTransportError
    return envelope


def _text_blocks(result: dict[str, Any]) -> list[str]:
    """Return every text block carried by an MCP tool result, in order."""
    blocks = []
    for block in result.get("content") or []:
        if not isinstance(block, dict):
            continue
        text = block.get("text")
        if isinstance(text, str):
            blocks.append(text)
    return blocks


def _tool_error_message(result: dict[str, Any]) -> str | None:
    """Return the human-readable message from an `isError` tool result.

    MCP puts the failure text in the result's `content` blocks rather than in
    a structured error object, so surface the first text block; Swiggy's
    messages are user-facing (e.g. "currently available only for beta users").
    """
    blocks = _text_blocks(result)
    return blocks[0] if blocks else None


def _swiggy_envelope(result: dict[str, Any]) -> dict[str, Any] | None:
    """Return Swiggy's `{success, data, message}` envelope from a tool result.

    Swiggy does not populate MCP's own `structuredContent`; it serializes its
    envelope as JSON *text* inside the first `content` block, so the real
    payload arrives double-encoded. Returns the first text block that decodes
    to a JSON object, or None if no block does (e.g. a tool that answers with
    prose rather than an envelope).
    """
    for text in _text_blocks(result):
        try:
            decoded = json.loads(text)
        except ValueError:
            continue
        if isinstance(decoded, dict):
            return decoded
    return None


def _envelope_error_message(envelope: dict[str, Any]) -> str | None:
    """Return the failure text from a `success: false` Swiggy envelope.

    Per https://mcp.swiggy.com/builders/docs/reference/errors.md the message
    lives at `error.message`; fall back to a top-level `message` so a tool
    that flags failure without the documented `error` object still surfaces
    something actionable rather than a bare `None`.
    """
    error = envelope.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str):
            return message
    message = envelope.get("message")
    return message if isinstance(message, str) else None


async def call_tool(
    access_token: str, tool_name: str, arguments: dict[str, Any]
) -> Any:
    """Invoke one Instamart MCP tool and return its `data` payload.

    Args:
        access_token: The caller's live Swiggy access token.
        tool_name: The MCP tool to invoke (e.g. "get_addresses").
        arguments: The tool's JSON-RPC arguments.

    Swiggy wraps every tool payload in its own `{success, data, message}`
    envelope and serializes that envelope as JSON text inside the MCP result's
    first `content` block - it does not use MCP's `structuredContent` field.
    This unwraps both layers so callers only ever see `data`.

    Returns:
        The tool's `data` payload (shape is tool-specific), or None if the
        tool returned neither `structuredContent` nor a JSON text envelope.

    Raises:
        InstamartAuthError: Swiggy rejected the token (HTTP 401/419, or
            JSON-RPC error -32001).
        InstamartTransportError: The request failed at the network, HTTP, or
            JSON-RPC protocol level before a tool-level result was reached.
        InstamartDomainError: The tool ran but reported failure - either
            `success: false` in Swiggy's envelope (the signal that actually
            fires in practice) or MCP's own `isError` flag. Terminal, not
            retryable.
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
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Accept": MCP_ACCEPT_HEADER,
                },
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

    payload = _decode_envelope(response)
    rpc_error = payload.get("error")
    if rpc_error is not None:
        if rpc_error.get("code") == JSONRPC_AUTH_ERROR_CODE:
            raise InstamartAuthError
        logger.warning("Instamart tool %s JSON-RPC error: %s", tool_name, rpc_error)
        raise InstamartTransportError(rpc_error.get("message"))

    result = payload.get("result") or {}
    # MCP's own tool-level failure flag. Swiggy does not appear to set it in
    # practice (its failures come back as `success: false` below), but it is
    # the protocol-level contract, so honour it if it ever does surface.
    if result.get("isError"):
        raise InstamartDomainError(_tool_error_message(result))

    # Forward-compatible: prefer MCP's structured field if Swiggy ever starts
    # populating it. As of CUE-77 it is absent from every observed response.
    structured = result.get("structuredContent")
    if structured is not None:
        return structured

    # The real wire format: Swiggy's `{success, data, message}` envelope, JSON
    # encoded into a text content block.
    envelope = _swiggy_envelope(result)
    if envelope is None:
        logger.warning(
            "Instamart tool %s returned no structured or JSON-text payload",
            tool_name,
        )
        return None

    if envelope.get("success") is False:
        raise InstamartDomainError(_envelope_error_message(envelope))

    return envelope.get("data")
