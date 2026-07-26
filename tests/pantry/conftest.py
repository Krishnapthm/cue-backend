from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator

import httpx
import pytest_asyncio
from httpx import ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import current_user
from app.database import get_session
from app.main import app
from app.models.cart import CartPlan
from app.models.chat import ChatSession
from app.models.user import User
from app.pantry.constants import PantryCategory
from app.pantry.models import PantryItem


@pytest_asyncio.fixture
async def other_user(db_session: AsyncSession) -> User:
    """A second, distinct Cue user - used to prove cross-user isolation."""
    new_user = User(
        firebase_uid=f"firebase-uid-{uuid.uuid4()}", email="other@example.com"
    )
    db_session.add(new_user)
    await db_session.commit()
    await db_session.refresh(new_user)
    return new_user


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
    client: httpx.AsyncClient, db_session: AsyncSession, user: User
) -> AsyncGenerator[httpx.AsyncClient]:
    """`client`, signed in as `user`.

    Only auth is faked - the router, service and real Postgres schema are all
    exercised for real.

    The override re-loads the user per request rather than closing over the
    instance, mirroring what the real `current_user` does. Handing back a
    long-lived ORM object instead would break as soon as a test triggers a
    rollback: that expires the instance, and the next attribute read inside a
    route would attempt lazy IO from async context.
    """
    user_id = user.id

    async def override_current_user() -> User:
        loaded = await db_session.get(User, user_id)
        assert loaded is not None
        return loaded

    app.dependency_overrides[current_user] = override_current_user
    yield client


@pytest_asyncio.fixture
async def chat_session(db_session: AsyncSession, user: User) -> ChatSession:
    """A chat session owned by `user`, to hang a `CartPlan` off of."""
    session_row = ChatSession(user_id=user.id)
    db_session.add(session_row)
    await db_session.commit()
    await db_session.refresh(session_row)
    return session_row


@pytest_asyncio.fixture
async def cart_plan(db_session: AsyncSession, chat_session: ChatSession) -> CartPlan:
    """A live `CartPlan`, to hang the ingredient rows a stamp matches against."""
    plan = CartPlan(session_id=chat_session.id, address_id="addr-1")
    db_session.add(plan)
    await db_session.commit()
    await db_session.refresh(plan)
    return plan


@pytest_asyncio.fixture
async def pantry_item(db_session: AsyncSession, user: User) -> PantryItem:
    """One full jar of basmati rice in `user`'s pantry."""
    item = PantryItem(
        user_id=user.id,
        name="Basmati Rice",
        name_normalized="basmati rice",
        category=PantryCategory.GRAINS_AND_PULSES.value,
        level=3,
    )
    db_session.add(item)
    await db_session.commit()
    await db_session.refresh(item)
    return item
