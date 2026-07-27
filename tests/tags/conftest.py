from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import current_user
from app.database import get_session
from app.main import app
from app.models.provider import ProviderLink
from app.models.user import User
from app.providers import service as provider_service
from app.providers.constants import PROVIDER
from tests.conftest import INSTAMART_ACCESS_TOKEN, InstamartToolCallStub

ADDRESS_ID = "addr-1"


def product(
    *,
    product_id: str,
    name: str,
    spin_id: str,
    price: str,
    pack_size: str | None = "1 kg",
    in_stock: bool = True,
    brand: str | None = None,
) -> dict[str, Any]:
    """One `search_products` candidate, in Swiggy's wire shape."""
    return {
        "productId": product_id,
        "name": name,
        "brand": brand,
        "variants": [
            {
                "spinId": spin_id,
                "packSize": pack_size,
                "price": price,
                "inStock": in_stock,
            }
        ],
    }


def search_result(*products: dict[str, Any]) -> dict[str, Any]:
    """A `search_products` tool result carrying `products`."""
    return {"structuredContent": {"products": list(products)}}


def go_to_result(*items: dict[str, Any]) -> dict[str, Any]:
    """A `your_go_to_items` tool result carrying `items`."""
    return {"structuredContent": {"items": list(items)}}


def go_to_item(
    *, product_name: str, brand: str, spin_id: str = "spin-go-to"
) -> dict[str, Any]:
    """One previously-ordered product, in Swiggy's wire shape (CUE-74).

    Mirrors an observed live `your_go_to_items` response: `displayName` and
    `variations`, not `productName`/`variants`, and a real top-level `brand`.
    """
    return {
        "displayName": product_name,
        "brand": brand,
        "variations": [{"spinId": spin_id}],
    }


@pytest.fixture
def instamart(mock_instamart_tool_call: InstamartToolCallStub) -> InstamartToolCallStub:
    """The Swiggy MCP stub, defaulting to an empty go-to list and no results."""
    mock_instamart_tool_call.configure(result={"structuredContent": {}})
    return mock_instamart_tool_call


@pytest_asyncio.fixture
async def other_user(db_session: AsyncSession) -> User:
    """A second Cue user - used to prove bindings never cross users."""
    new_user = User(
        firebase_uid=f"firebase-uid-{uuid.uuid4()}", email="other-tags@example.com"
    )
    db_session.add(new_user)
    await db_session.commit()
    await db_session.refresh(new_user)
    return new_user


@pytest_asyncio.fixture
async def linked_other_user(db_session: AsyncSession, other_user: User) -> User:
    """`other_user`, with a live Swiggy link of their own."""
    db_session.add(
        ProviderLink(
            user_id=other_user.id,
            provider=PROVIDER,
            access_token_ct=provider_service._encrypt(INSTAMART_ACCESS_TOKEN),
            token_expires_at=datetime.now(UTC) + timedelta(days=5),
            scope="mcp:tools",
            status="active",
        )
    )
    await db_session.commit()
    return other_user


@pytest_asyncio.fixture
async def client(db_session: AsyncSession) -> AsyncGenerator[httpx.AsyncClient]:
    """An unauthenticated client bound to the ephemeral test database."""

    async def override_get_session() -> AsyncGenerator[AsyncSession]:
        yield db_session

    app.dependency_overrides[get_session] = override_get_session
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", follow_redirects=False
    ) as ac:
        yield ac
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def authed_client(
    client: httpx.AsyncClient, db_session: AsyncSession, linked_user: User
) -> AsyncGenerator[httpx.AsyncClient]:
    """`client`, signed in as a user with a live Swiggy link.

    The override re-loads the user per request rather than closing over the
    instance: a rollback anywhere in a test expires the fixture object, and
    the next attribute read inside a route would attempt lazy IO from async
    context.
    """
    user_id = linked_user.id

    async def override_current_user() -> User:
        loaded = await db_session.get(User, user_id)
        assert loaded is not None
        return loaded

    app.dependency_overrides[current_user] = override_current_user
    yield client
