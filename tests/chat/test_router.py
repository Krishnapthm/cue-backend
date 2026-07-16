from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncGenerator

import httpx
import pytest_asyncio
from httpx import ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.main import app
from app.models.chat import ChatSession
from app.models.user import User


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
    client: httpx.AsyncClient, user: User
) -> AsyncGenerator[httpx.AsyncClient]:
    from app.auth.dependencies import current_user

    async def override_current_user() -> User:
        return user

    app.dependency_overrides[current_user] = override_current_user
    yield client


async def test_create_requires_authentication(client: httpx.AsyncClient) -> None:
    response = await client.post("/chat/sessions")

    assert response.status_code == 401


async def test_create_returns_a_new_untitled_session(
    authed_client: httpx.AsyncClient,
) -> None:
    response = await authed_client.post("/chat/sessions")

    assert response.status_code == 201
    body = response.json()
    assert body["title"] is None
    uuid.UUID(body["id"])
    assert "updated_at" in body


async def test_list_recents_requires_authentication(client: httpx.AsyncClient) -> None:
    response = await client.get("/chat/sessions")

    assert response.status_code == 401


async def test_list_recents_returns_only_the_callers_sessions(
    authed_client: httpx.AsyncClient,
    db_session: AsyncSession,
    other_user: User,
) -> None:
    mine = await authed_client.post("/chat/sessions")
    other_session = ChatSession(user_id=other_user.id)
    db_session.add(other_session)
    await db_session.commit()

    response = await authed_client.get("/chat/sessions")

    assert response.status_code == 200
    ids = [s["id"] for s in response.json()]
    assert ids == [mine.json()["id"]]


async def test_list_recents_orders_most_recently_updated_first(
    authed_client: httpx.AsyncClient,
) -> None:
    first = await authed_client.post("/chat/sessions")
    await asyncio.sleep(0.01)
    second = await authed_client.post("/chat/sessions")

    response = await authed_client.get("/chat/sessions")

    ids = [s["id"] for s in response.json()]
    assert ids == [second.json()["id"], first.json()["id"]]


async def test_get_requires_authentication(client: httpx.AsyncClient) -> None:
    response = await client.get(f"/chat/sessions/{uuid.uuid4()}")

    assert response.status_code == 401


async def test_get_returns_404_for_an_unknown_session(
    authed_client: httpx.AsyncClient,
) -> None:
    response = await authed_client.get(f"/chat/sessions/{uuid.uuid4()}")

    assert response.status_code == 404


async def test_get_returns_404_for_another_users_session(
    authed_client: httpx.AsyncClient,
    db_session: AsyncSession,
    other_user: User,
) -> None:
    other_session = ChatSession(user_id=other_user.id)
    db_session.add(other_session)
    await db_session.commit()
    await db_session.refresh(other_session)

    response = await authed_client.get(f"/chat/sessions/{other_session.id}")

    assert response.status_code == 404


async def test_get_returns_the_session_with_its_ordered_transcript(
    authed_client: httpx.AsyncClient,
) -> None:
    create_response = await authed_client.post("/chat/sessions")
    session_id = create_response.json()["id"]
    await authed_client.post(
        f"/chat/sessions/{session_id}/messages",
        json={"role": "user", "content": "one"},
    )
    await authed_client.post(
        f"/chat/sessions/{session_id}/messages",
        json={"role": "assistant", "content": "two"},
    )

    response = await authed_client.get(f"/chat/sessions/{session_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == session_id
    assert body["title"] is None
    assert body["selected_address_id"] is None
    assert [m["content"] for m in body["messages"]] == ["one", "two"]


async def test_add_message_requires_authentication(client: httpx.AsyncClient) -> None:
    response = await client.post(
        f"/chat/sessions/{uuid.uuid4()}/messages",
        json={"role": "user", "content": "hi"},
    )

    assert response.status_code == 401


async def test_add_message_returns_404_for_an_unknown_session(
    authed_client: httpx.AsyncClient,
) -> None:
    response = await authed_client.post(
        f"/chat/sessions/{uuid.uuid4()}/messages",
        json={"role": "user", "content": "hi"},
    )

    assert response.status_code == 404


async def test_add_message_returns_404_for_another_users_session(
    authed_client: httpx.AsyncClient,
    db_session: AsyncSession,
    other_user: User,
) -> None:
    other_session = ChatSession(user_id=other_user.id)
    db_session.add(other_session)
    await db_session.commit()
    await db_session.refresh(other_session)

    response = await authed_client.post(
        f"/chat/sessions/{other_session.id}/messages",
        json={"role": "user", "content": "hi"},
    )

    assert response.status_code == 404


async def test_add_message_persists_a_text_message(
    authed_client: httpx.AsyncClient,
) -> None:
    create_response = await authed_client.post("/chat/sessions")
    session_id = create_response.json()["id"]

    response = await authed_client.post(
        f"/chat/sessions/{session_id}/messages",
        json={"role": "user", "content": "What's for dinner?"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["role"] == "user"
    assert body["kind"] == "text"
    assert body["content"] == "What's for dinner?"
    assert body["payload"] is None
    assert isinstance(body["id"], int)


async def test_add_message_persists_a_non_text_message_with_payload(
    authed_client: httpx.AsyncClient,
) -> None:
    create_response = await authed_client.post("/chat/sessions")
    session_id = create_response.json()["id"]

    response = await authed_client.post(
        f"/chat/sessions/{session_id}/messages",
        json={
            "role": "assistant",
            "kind": "cart_ready",
            "payload": {"cart_id": "abc123"},
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["kind"] == "cart_ready"
    assert body["content"] is None
    assert body["payload"] == {"cart_id": "abc123"}


async def test_add_message_rejects_text_kind_without_content(
    authed_client: httpx.AsyncClient,
) -> None:
    create_response = await authed_client.post("/chat/sessions")
    session_id = create_response.json()["id"]

    response = await authed_client.post(
        f"/chat/sessions/{session_id}/messages",
        json={"role": "user", "kind": "text"},
    )

    assert response.status_code == 422


async def test_add_message_rejects_non_text_kind_without_payload(
    authed_client: httpx.AsyncClient,
) -> None:
    create_response = await authed_client.post("/chat/sessions")
    session_id = create_response.json()["id"]

    response = await authed_client.post(
        f"/chat/sessions/{session_id}/messages",
        json={"role": "assistant", "kind": "image"},
    )

    assert response.status_code == 422


async def test_add_message_resurfaces_session_at_the_top_of_recents(
    authed_client: httpx.AsyncClient,
) -> None:
    older = await authed_client.post("/chat/sessions")
    await asyncio.sleep(0.01)
    newer = await authed_client.post("/chat/sessions")
    recents = await authed_client.get("/chat/sessions")
    assert [s["id"] for s in recents.json()] == [
        newer.json()["id"],
        older.json()["id"],
    ]

    await authed_client.post(
        f"/chat/sessions/{older.json()['id']}/messages",
        json={"role": "user", "content": "hi"},
    )

    recents_after = await authed_client.get("/chat/sessions")
    assert [s["id"] for s in recents_after.json()] == [
        older.json()["id"],
        newer.json()["id"],
    ]
