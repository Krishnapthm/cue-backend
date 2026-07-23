from __future__ import annotations

from collections.abc import AsyncGenerator, Generator

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.main import app
from app.models.user import User
from app.orders.service import _reset_tracking_cache


@pytest.fixture(autouse=True)
def reset_tracking_cache() -> Generator[None]:
    """Clear the process-local tracking cache before and after each test.

    The cache is a module-level dict keyed on (user_id, order_id); without
    this, state from one test's poll-floor assertions could leak into the
    next test's timing window.
    """
    _reset_tracking_cache()
    yield
    _reset_tracking_cache()


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
    """`client`, authenticated as `linked_user` - a Swiggy link is required
    or `_call_authenticated` raises `InstamartAuthError` (-> 401) before the
    route's own logic ever runs."""
    from app.auth.dependencies import current_user

    async def override_current_user() -> User:
        return linked_user

    app.dependency_overrides[current_user] = override_current_user
    yield client
