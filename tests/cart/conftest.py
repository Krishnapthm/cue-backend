from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.instamart.constants import TOOL_GET_CART, TOOL_UPDATE_CART
from app.instamart.exceptions import InstamartDomainError
from app.main import app
from app.models.cart import CartPlan
from app.models.chat import ChatSession
from app.models.user import User

# Every fake line is priced the same, so a cart total is just a count - the
# cart endpoints never reason about price, only about which lines survive.
FAKE_UNIT_PRICE = Decimal("50.00")


@dataclass
class FakeInstamart:
    """An in-memory Swiggy that implements `update_cart`'s replace semantics.

    The shared `mock_instamart_tool_call` stub answers per tool *name*, which
    cannot express "the third `update_cart` call behaves differently from the
    first" - exactly what the per-item retry path needs. This fake holds real
    cart state instead, so the tests assert against what the user's cart
    actually ends up being rather than against a canned payload.

    Attributes:
        items: The current cart, `spin_id` -> quantity, insertion-ordered.
        rejects: Spin ids Swiggy refuses outright - any `update_cart`
            including one fails the whole call, as the real tool does.
        drops: Spin ids Swiggy accepts and then silently omits from the cart
            it returns (`success: true`, item quietly gone).
        metadata: Extra per-spin fields Swiggy includes in cart line read-backs.
        writes: Every `update_cart` argument list, for asserting call counts.
    """

    items: dict[str, int] = field(default_factory=dict)
    rejects: set[str] = field(default_factory=set)
    drops: set[str] = field(default_factory=set)
    metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    writes: list[list[dict[str, Any]]] = field(default_factory=list)

    def _cart_payload(self) -> dict[str, Any]:
        return {
            "cart": {
                "items": [
                    {
                        "spinId": spin_id,
                        "quantity": quantity,
                        "price": str(FAKE_UNIT_PRICE * quantity),
                        **self.metadata.get(spin_id, {}),
                    }
                    for spin_id, quantity in self.items.items()
                ],
                "total": str(FAKE_UNIT_PRICE * sum(self.items.values())),
                "availablePaymentMethods": ["COD"],
            }
        }

    async def call_tool(
        self, _access_token: str, tool_name: str, arguments: dict[str, Any]
    ) -> Any:
        # A real MCP call awaits the network here. Yielding to the event loop
        # reproduces that suspension point, which is what lets two concurrent
        # requests interleave - without it the concurrency test would pass
        # against an unlocked read-merge-write and prove nothing.
        await asyncio.sleep(0)
        if tool_name == TOOL_GET_CART:
            return self._cart_payload()
        if tool_name == TOOL_UPDATE_CART:
            requested = arguments["items"]
            self.writes.append(requested)
            for item in requested:
                if item["spinId"] in self.rejects:
                    raise InstamartDomainError(
                        f"{item['spinId']} is out of stock at this address."
                    )
            self.items = {
                item["spinId"]: item["quantity"]
                for item in requested
                if item["spinId"] not in self.drops
            }
            return self._cart_payload()
        raise AssertionError(f"Unexpected tool call: {tool_name}")


@pytest.fixture
def fake_instamart(monkeypatch: pytest.MonkeyPatch) -> FakeInstamart:
    """Replace the MCP tool call with a stateful in-memory Swiggy cart."""
    fake = FakeInstamart()
    monkeypatch.setattr("app.instamart.client.call_tool", fake.call_tool)
    return fake


@pytest_asyncio.fixture
async def client(db_session: AsyncSession) -> AsyncGenerator[httpx.AsyncClient]:
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
    client: httpx.AsyncClient, linked_user: User
) -> AsyncGenerator[httpx.AsyncClient]:
    """`client`, authenticated as `linked_user`.

    A Swiggy link is required or `_call_authenticated` raises
    `InstamartAuthError` (-> 401) before the route's own logic ever runs.
    """
    from app.auth.dependencies import current_user

    async def override_current_user() -> User:
        return linked_user

    app.dependency_overrides[current_user] = override_current_user
    yield client


@pytest_asyncio.fixture
async def chat_session(db_session: AsyncSession, user: User) -> ChatSession:
    """A chat session owned by `user`, to hang a `CartPlan` off of.

    Depends on the plain `user` fixture, not `linked_user`: a test asserting
    "not linked" behavior must be able to request `user` and `chat_session`
    together without `linked_user` ever being instantiated (fixtures are
    cached per test by name, so pulling in `linked_user` anywhere in the
    graph would silently link this same user).
    """
    session_row = ChatSession(user_id=user.id)
    db_session.add(session_row)
    await db_session.commit()
    await db_session.refresh(session_row)
    return session_row


@pytest_asyncio.fixture
async def cart_plan(db_session: AsyncSession, chat_session: ChatSession) -> CartPlan:
    """A live (non-superseded) `CartPlan` for `chat_session`, to check out."""
    plan = CartPlan(session_id=chat_session.id, address_id="addr-1")
    db_session.add(plan)
    await db_session.commit()
    await db_session.refresh(plan)
    return plan
