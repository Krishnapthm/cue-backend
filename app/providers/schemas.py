from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class ProviderStatus(StrEnum):
    """Swiggy link state surfaced to Settings and the chat prompt (R9.2)."""

    CONNECTED = "connected"
    RECONNECT_NEEDED = "reconnect_needed"
    NOT_CONNECTED = "not_connected"


class AuthorizeResponse(BaseModel):
    """The Swiggy consent URL the client should open to link the account."""

    authorize_url: str


class StatusResponse(BaseModel):
    """Current Swiggy link state for the Cue user."""

    status: ProviderStatus
