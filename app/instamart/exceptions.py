from __future__ import annotations

from fastapi import status

from app.exceptions import AppError


class InstamartAuthError(AppError):
    """Swiggy rejected the access token, or the session was revoked (R2.5).

    Callers must route this through the provider recovery ladder
    (`app.providers.service.mark_link_expired`), not retry the call: Swiggy
    MCP v1.0 issues no refresh token, so the only recovery is a fresh OAuth
    authorize.
    """

    status_code = status.HTTP_401_UNAUTHORIZED
    detail = "Swiggy session expired; reconnect required."


class InstamartTransportError(AppError):
    """The Instamart MCP call failed before a tool-level result was reached.

    Covers network failures, HTTP 4xx/5xx, and JSON-RPC protocol errors other
    than auth. Per the Swiggy error reference, upstream timeouts/5xx are
    retryable with backoff; malformed requests are not. Callers decide which
    applies for their tool.
    """

    status_code = status.HTTP_502_BAD_GATEWAY
    detail = "Failed to reach Swiggy Instamart."


class InstamartDomainError(AppError):
    """The tool executed but Swiggy reported `success: false`.

    Terminal per the Swiggy error reference (e.g. out of stock, store
    closed): not retryable, the message is meant to be surfaced to the user.
    """

    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    detail = "Swiggy Instamart could not complete the request."
