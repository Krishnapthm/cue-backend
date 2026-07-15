"""get_addresses / create_address / delete_address wrappers (CUE-10).

Every call resolves the user's live Swiggy access token first (R2.5): no
usable token means "not linked or reconnect needed", which is routed through
`InstamartAuthError` exactly like a live auth failure - never a raw failure.
A live 401/419 from Swiggy additionally marks the link expired so the next
call short-circuits the same way instead of re-hitting Swiggy with a token
we now know is dead.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.instamart import client
from app.instamart.constants import (
    TOOL_CREATE_ADDRESS,
    TOOL_DELETE_ADDRESS,
    TOOL_GET_ADDRESSES,
)
from app.instamart.exceptions import InstamartAuthError
from app.instamart.schemas import Address, CreateAddressRequest
from app.providers import service as provider_service


async def _call_authenticated(
    session: AsyncSession, user_id: int, tool_name: str, arguments: dict[str, Any]
) -> Any:
    """Call an Instamart tool with the user's token, applying the recovery ladder."""
    token = await provider_service.get_decrypted_access_token(session, user_id)
    if token is None:
        raise InstamartAuthError
    try:
        return await client.call_tool(token, tool_name, arguments)
    except InstamartAuthError:
        await provider_service.mark_link_expired(session, user_id)
        raise


async def get_addresses(session: AsyncSession, user_id: int) -> list[Address]:
    """Return the user's saved Swiggy delivery addresses (R3.1 smoke test)."""
    data = await _call_authenticated(session, user_id, TOOL_GET_ADDRESSES, {})
    # The envelope key holding the list isn't pinned by Swiggy's docs; accept
    # either a bare list or one nested under "addresses".
    raw_addresses = data.get("addresses", []) if isinstance(data, dict) else data or []
    return [Address.model_validate(item) for item in raw_addresses]


async def create_address(
    session: AsyncSession, user_id: int, request: CreateAddressRequest
) -> Address:
    """Create a new Swiggy delivery address for the user (R3.1)."""
    arguments = request.model_dump(by_alias=True, exclude_none=True)
    data = await _call_authenticated(session, user_id, TOOL_CREATE_ADDRESS, arguments)
    # Same envelope ambiguity as get_addresses: accept the address nested
    # under "address" or returned as the data payload directly.
    raw_address = data.get("address", data) if isinstance(data, dict) else data
    return Address.model_validate(raw_address)


async def delete_address(session: AsyncSession, user_id: int, address_id: str) -> None:
    """Delete a saved Swiggy delivery address.

    Irreversible on Swiggy's side; confirming with the user before calling
    this is the caller's responsibility.
    """
    await _call_authenticated(
        session, user_id, TOOL_DELETE_ADDRESS, {"addressId": address_id}
    )
