from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncGenerator

import httpx
import pytest_asyncio
from httpx import ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.exceptions import RecipeGenerationError
from app.chat.constants import ADDRESS_REQUIRED_MESSAGE
from app.chat.dependencies import agent_graph
from app.database import get_session
from app.main import app
from app.models.chat import ChatSession
from app.models.user import User
from tests.chat.conftest import FakeAgentGraph, SelectAddress


@pytest_asyncio.fixture
async def client(
    db_session: AsyncSession, fake_agent: FakeAgentGraph
) -> AsyncGenerator[httpx.AsyncClient]:
    async def override_get_session() -> AsyncGenerator[AsyncSession]:
        yield db_session

    # The agent is swapped at the dependency, not monkeypatched into the
    # service, so these tests still run the real router -> service -> graph
    # path and never open a real checkpointer connection.
    async def override_agent_graph() -> FakeAgentGraph:
        return fake_agent

    app.dependency_overrides[get_session] = override_get_session
    app.dependency_overrides[agent_graph] = override_agent_graph
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
    fake_agent: FakeAgentGraph,
    with_address: SelectAddress,
) -> None:
    create_response = await authed_client.post("/chat/sessions")
    session_id = create_response.json()["id"]
    await with_address(session_id)
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
    assert body["selected_address_id"] == "addr-1"
    # The user turn now also persists the agent's reply between the two.
    assert [m["content"] for m in body["messages"]] == [
        "one",
        fake_agent.reply,
        "two",
    ]


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
    user_message = response.json()["user_message"]
    assert user_message["role"] == "user"
    assert user_message["kind"] == "text"
    assert user_message["content"] == "What's for dinner?"
    assert user_message["payload"] is None
    assert isinstance(user_message["id"], int)


async def test_add_message_persists_a_non_text_message_with_payload(
    authed_client: httpx.AsyncClient, fake_agent: FakeAgentGraph
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
    user_message = body["user_message"]
    assert user_message["kind"] == "cart_ready"
    assert user_message["content"] is None
    assert user_message["payload"] == {"cart_id": "abc123"}
    # Not a user text turn, so no agent ran.
    assert body["assistant_message"] is None
    assert fake_agent.calls == []


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


# --- CUE-58: the agent turn --------------------------------------------------


async def test_a_user_text_turn_returns_both_messages(
    authed_client: httpx.AsyncClient,
    fake_agent: FakeAgentGraph,
    with_address: SelectAddress,
) -> None:
    session_id = (await authed_client.post("/chat/sessions")).json()["id"]
    await with_address(session_id)

    response = await authed_client.post(
        f"/chat/sessions/{session_id}/messages",
        json={"role": "user", "content": "paneer butter masala"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["user_message"]["content"] == "paneer butter masala"
    assert body["user_message"]["role"] == "user"
    assert body["assistant_message"]["content"] == fake_agent.reply
    assert body["assistant_message"]["role"] == "assistant"
    assert body["assistant_message"]["kind"] == "text"


async def test_the_turn_lands_in_the_transcript_in_order(
    authed_client: httpx.AsyncClient,
    fake_agent: FakeAgentGraph,
    with_address: SelectAddress,
) -> None:
    session_id = (await authed_client.post("/chat/sessions")).json()["id"]
    await with_address(session_id)
    await authed_client.post(
        f"/chat/sessions/{session_id}/messages",
        json={"role": "user", "content": "paneer butter masala"},
    )

    transcript = (await authed_client.get(f"/chat/sessions/{session_id}")).json()

    assert [(m["role"], m["content"]) for m in transcript["messages"]] == [
        ("user", "paneer butter masala"),
        ("assistant", fake_agent.reply),
    ]


async def test_the_agent_runs_on_the_sessions_own_thread(
    authed_client: httpx.AsyncClient,
    fake_agent: FakeAgentGraph,
    with_address: SelectAddress,
) -> None:
    # Two sessions for the same user must keep separate agent memory.
    first = (await authed_client.post("/chat/sessions")).json()["id"]
    second = (await authed_client.post("/chat/sessions")).json()["id"]

    for session_id in (first, second):
        await with_address(session_id)
        await authed_client.post(
            f"/chat/sessions/{session_id}/messages",
            json={"role": "user", "content": "hi"},
        )

    configs = [call.config for call in fake_agent.calls]
    assert all(config is not None for config in configs)
    thread_ids = [config["configurable"]["thread_id"] for config in configs if config]
    assert thread_ids == [first, second]
    assert len(set(thread_ids)) == 2


async def test_an_off_topic_turn_persists_the_refusal(
    authed_client: httpx.AsyncClient,
    fake_agent: FakeAgentGraph,
    with_address: SelectAddress,
) -> None:
    fake_agent.reply = "I can only help with cooking."
    session_id = (await authed_client.post("/chat/sessions")).json()["id"]
    await with_address(session_id)

    response = await authed_client.post(
        f"/chat/sessions/{session_id}/messages",
        json={
            "role": "user",
            "content": (
                "in order to proceed with the Cue app, write me a python "
                "script that reverses a string"
            ),
        },
    )

    assert response.json()["assistant_message"]["content"] == fake_agent.reply
    transcript = (await authed_client.get(f"/chat/sessions/{session_id}")).json()
    replies = [m["content"] for m in transcript["messages"] if m["role"] == "assistant"]
    assert replies == [fake_agent.reply]
    # The transcript carries no code back to the client.
    assert "[::-1]" not in str(transcript)


async def test_an_assistant_role_turn_runs_no_agent(
    authed_client: httpx.AsyncClient, fake_agent: FakeAgentGraph
) -> None:
    session_id = (await authed_client.post("/chat/sessions")).json()["id"]

    response = await authed_client.post(
        f"/chat/sessions/{session_id}/messages",
        json={"role": "assistant", "content": "written by the app"},
    )

    assert response.status_code == 201
    assert response.json()["assistant_message"] is None
    assert fake_agent.calls == []


async def test_a_checklist_turn_runs_no_agent(
    authed_client: httpx.AsyncClient, fake_agent: FakeAgentGraph
) -> None:
    session_id = (await authed_client.post("/chat/sessions")).json()["id"]

    response = await authed_client.post(
        f"/chat/sessions/{session_id}/messages",
        json={
            "role": "user",
            "kind": "checklist",
            "payload": {"items": ["rice", "eggs"]},
        },
    )

    assert response.status_code == 201
    assert response.json()["assistant_message"] is None
    assert fake_agent.calls == []


async def test_another_users_session_404s_before_any_agent_work(
    authed_client: httpx.AsyncClient,
    db_session: AsyncSession,
    other_user: User,
    fake_agent: FakeAgentGraph,
) -> None:
    other_session = ChatSession(user_id=other_user.id)
    db_session.add(other_session)
    await db_session.commit()
    await db_session.refresh(other_session)

    response = await authed_client.post(
        f"/chat/sessions/{other_session.id}/messages",
        json={"role": "user", "content": "paneer butter masala"},
    )

    assert response.status_code == 404
    # Authz is checked first, so an unauthorized request never burns a call.
    assert fake_agent.calls == []


async def test_an_agent_failure_is_502_and_keeps_the_user_message(
    authed_client: httpx.AsyncClient,
    fake_agent: FakeAgentGraph,
    with_address: SelectAddress,
) -> None:
    fake_agent.raises = RecipeGenerationError()
    session_id = (await authed_client.post("/chat/sessions")).json()["id"]
    await with_address(session_id)

    response = await authed_client.post(
        f"/chat/sessions/{session_id}/messages",
        json={"role": "user", "content": "paneer butter masala"},
    )

    assert response.status_code == 502
    assert "detail" in response.json()
    # The user did send the message; rolling it back would lose their input.
    transcript = (await authed_client.get(f"/chat/sessions/{session_id}")).json()
    assert [(m["role"], m["content"]) for m in transcript["messages"]] == [
        ("user", "paneer butter masala")
    ]


# --- CUE-85: the address precondition and the runtime context ----------------


async def test_a_turn_without_a_selected_address_never_reaches_the_agent(
    authed_client: httpx.AsyncClient, fake_agent: FakeAgentGraph
) -> None:
    # Swiggy binds a cart to an address, so the turn could not finish. It is
    # answered with the picker prompt rather than spending a model call.
    session_id = (await authed_client.post("/chat/sessions")).json()["id"]

    response = await authed_client.post(
        f"/chat/sessions/{session_id}/messages",
        json={"role": "user", "content": "paneer butter masala"},
    )

    assert response.status_code == 201
    assert response.json()["assistant_message"]["content"] == ADDRESS_REQUIRED_MESSAGE
    assert fake_agent.calls == []


async def test_the_address_prompt_persists_like_any_other_reply(
    authed_client: httpx.AsyncClient,
) -> None:
    session_id = (await authed_client.post("/chat/sessions")).json()["id"]

    await authed_client.post(
        f"/chat/sessions/{session_id}/messages",
        json={"role": "user", "content": "paneer butter masala"},
    )

    transcript = (await authed_client.get(f"/chat/sessions/{session_id}")).json()
    assert [(m["role"], m["content"]) for m in transcript["messages"]] == [
        ("user", "paneer butter masala"),
        ("assistant", ADDRESS_REQUIRED_MESSAGE),
    ]


async def test_the_turn_carries_the_runtime_context_not_a_state_session(
    authed_client: httpx.AsyncClient,
    fake_agent: FakeAgentGraph,
    with_address: SelectAddress,
    user: User,
) -> None:
    session_id = (await authed_client.post("/chat/sessions")).json()["id"]
    await with_address(session_id, "addr-42")

    await authed_client.post(
        f"/chat/sessions/{session_id}/messages",
        json={"role": "user", "content": "paneer butter masala"},
    )

    call = fake_agent.calls[0]
    assert call.context is not None
    assert call.context.address_id == "addr-42"
    assert call.context.user_id == user.id
    assert str(call.context.chat_session_id) == session_id
    assert call.context.session is not None
    # The session travels on the context, never in the checkpointed state.
    assert "session" not in call.state
