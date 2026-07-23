from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class ProviderStatus(StrEnum):
    """Swiggy link state surfaced to Settings and the chat prompt (R9.2)."""

    CONNECTED = "connected"
    RECONNECT_NEEDED = "reconnect_needed"
    NOT_CONNECTED = "not_connected"


class AuthorizeResponse(BaseModel):
    """The Swiggy consent URL the client should open to link the account."""

    authorize_url: str
    # The redirect URI registered with Swiggy, echoed back so the in-app
    # WebView knows which navigation to intercept (CUE-63/CUE-64). Served from
    # the backend so the client and `SWIGGY_REDIRECT_URI` cannot drift apart.
    redirect_uri: str


class CallbackRequest(BaseModel):
    """The `code`/`state` pair the client read off the intercepted redirect."""

    code: str = Field(min_length=1)
    state: str = Field(min_length=1)


class StatusResponse(BaseModel):
    """Current Swiggy link state for the Cue user."""

    status: ProviderStatus
