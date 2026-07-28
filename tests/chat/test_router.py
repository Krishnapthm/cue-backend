from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import httpx
import pytest_asyncio
from httpx import ASGITransport
from langgraph.types import Command
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.exceptions import RecipeGenerationError
from app.chat.constants import ADDRESS_REQUIRED_MESSAGE
from app.chat.dependencies import agent_graph
from app.database import get_session
from app.main import app
from app.models.chat import ChatSession
from app.models.user import User
from tests.chat.conftest import FakeAgentGraph, FakeInterrupt, SelectAddress


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


async def test_update_session_sets_the_selected_address(
    authed_client: httpx.AsyncClient,
) -> None:
    session_id = (await authed_client.post("/chat/sessions")).json()["id"]

    response = await authed_client.patch(
        f"/chat/sessions/{session_id}",
        json={"selected_address_id": "addr-1"},
    )

    assert response.status_code == 200
    assert response.json()["selected_address_id"] == "addr-1"
    detail = await authed_client.get(f"/chat/sessions/{session_id}")
    assert detail.json()["selected_address_id"] == "addr-1"


async def test_update_session_returns_404_for_another_users_session(
    authed_client: httpx.AsyncClient,
    db_session: AsyncSession,
    other_user: User,
) -> None:
    other_session = ChatSession(user_id=other_user.id)
    db_session.add(other_session)
    await db_session.commit()
    await db_session.refresh(other_session)

    response = await authed_client.patch(
        f"/chat/sessions/{other_session.id}",
        json={"selected_address_id": "addr-1"},
    )

    assert response.status_code == 404


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


async def test_an_unreadable_checklist_answer_is_422(
    authed_client: httpx.AsyncClient, fake_agent: FakeAgentGraph
) -> None:
    # A user `checklist` message is the resume value of the checklist interrupt
    # (CUE-90). A resume we cannot read is not consent: defaulting it to "none
    # of them" would silently buy everything the user already owns.
    session_id = (await authed_client.post("/chat/sessions")).json()["id"]

    response = await authed_client.post(
        f"/chat/sessions/{session_id}/messages",
        json={
            "role": "user",
            "kind": "checklist",
            "payload": {"items": ["rice", "eggs"]},
        },
    )

    assert response.status_code == 422
    assert fake_agent.calls == []


async def test_a_checklist_answer_with_nothing_paused_is_409(
    authed_client: httpx.AsyncClient,
    fake_agent: FakeAgentGraph,
    with_address: SelectAddress,
) -> None:
    # `Command(resume=...)` on a thread with no pending interrupt does not
    # error - it starts a fresh run and the session looks stuck. Failing loudly
    # here is what makes that impossible.
    session_id = (await authed_client.post("/chat/sessions")).json()["id"]
    await with_address(session_id)
    fake_agent.interrupts = ()

    response = await authed_client.post(
        f"/chat/sessions/{session_id}/messages",
        json={"role": "user", "kind": "checklist", "payload": {"have": ["salt"]}},
    )

    assert response.status_code == 409
    assert fake_agent.calls == []


async def test_a_cart_ready_turn_runs_no_agent(
    authed_client: httpx.AsyncClient, fake_agent: FakeAgentGraph
) -> None:
    session_id = (await authed_client.post("/chat/sessions")).json()["id"]

    response = await authed_client.post(
        f"/chat/sessions/{session_id}/messages",
        json={"role": "user", "kind": "cart_ready", "payload": {"plan_id": 1}},
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
    assert "session" not in call.turn_state


# --- the image intake path (CUE-88) ------------------------------------------


async def test_an_image_turn_reaches_the_agent_with_its_object_path(
    authed_client: httpx.AsyncClient,
    fake_agent: FakeAgentGraph,
    with_address: SelectAddress,
) -> None:
    # This is the `ChatMessage.payload` -> `AgentState` extraction CUE-23
    # deferred. `route_turn` branches on `image_object_path` alone, so it has to
    # arrive set or the photo path is unreachable.
    session_id = (await authed_client.post("/chat/sessions")).json()["id"]
    await with_address(session_id)

    response = await authed_client.post(
        f"/chat/sessions/{session_id}/messages",
        json={
            "role": "user",
            "kind": "image",
            "payload": {"object_path": "recipes/u1/8f2c.jpg"},
        },
    )

    assert response.status_code == 201
    assert fake_agent.calls[0].turn_state["image_object_path"] == "recipes/u1/8f2c.jpg"


async def test_a_text_turn_carries_no_image_path(
    authed_client: httpx.AsyncClient,
    fake_agent: FakeAgentGraph,
    with_address: SelectAddress,
) -> None:
    # The field survives in the checkpoint, so a text turn must actively clear
    # it - otherwise the turn after a photo would re-read the stale photo.
    session_id = (await authed_client.post("/chat/sessions")).json()["id"]
    await with_address(session_id)

    await authed_client.post(
        f"/chat/sessions/{session_id}/messages",
        json={"role": "user", "content": "paneer butter masala"},
    )

    assert fake_agent.calls[0].turn_state["image_object_path"] is None


async def test_an_image_turn_with_no_object_path_is_422(
    authed_client: httpx.AsyncClient, fake_agent: FakeAgentGraph
) -> None:
    # An unusable payload should fail at the boundary, not several seconds into
    # a turn the user is watching.
    session_id = (await authed_client.post("/chat/sessions")).json()["id"]

    response = await authed_client.post(
        f"/chat/sessions/{session_id}/messages",
        json={"role": "user", "kind": "image", "payload": {"url": "https://x/y.jpg"}},
    )

    assert response.status_code == 422
    assert fake_agent.calls == []


async def test_an_image_turn_with_a_blank_object_path_is_422(
    authed_client: httpx.AsyncClient, fake_agent: FakeAgentGraph
) -> None:
    session_id = (await authed_client.post("/chat/sessions")).json()["id"]

    response = await authed_client.post(
        f"/chat/sessions/{session_id}/messages",
        json={"role": "user", "kind": "image", "payload": {"object_path": ""}},
    )

    assert response.status_code == 422
    assert fake_agent.calls == []


# --- the checklist pause and its resume (CUE-90) ------------------------------


async def test_a_paused_turn_persists_the_checklist_and_no_reply(
    authed_client: httpx.AsyncClient,
    fake_agent: FakeAgentGraph,
    with_address: SelectAddress,
) -> None:
    # The turn owes the user a decision, not an answer. The checklist is
    # persisted so the transcript still renders it after a reconnect.
    session_id = (await authed_client.post("/chat/sessions")).json()["id"]
    await with_address(session_id)
    payload = {"ui": "checklist", "items": [{"name": "salt", "have": True}]}
    fake_agent.interrupt_value = payload

    response = await authed_client.post(
        f"/chat/sessions/{session_id}/messages",
        json={"role": "user", "content": "paneer butter masala"},
    )

    assert response.status_code == 201
    assistant = response.json()["assistant_message"]
    assert assistant["kind"] == "checklist"
    assert assistant["payload"] == payload


async def test_a_checklist_answer_resumes_the_same_thread(
    authed_client: httpx.AsyncClient,
    fake_agent: FakeAgentGraph,
    with_address: SelectAddress,
) -> None:
    session_id = (await authed_client.post("/chat/sessions")).json()["id"]
    await with_address(session_id)
    fake_agent.interrupts = (FakeInterrupt(id="int-1", value={"ui": "checklist"}),)

    response = await authed_client.post(
        f"/chat/sessions/{session_id}/messages",
        json={
            "role": "user",
            "kind": "checklist",
            "payload": {"have": ["salt", "pepper"]},
        },
    )

    assert response.status_code == 201
    call = fake_agent.calls[0]
    # A `Command(resume=...)`, never a plain state dict: a dict would not error,
    # it would start a fresh run and the session would look stuck.
    assert isinstance(call.state, Command)
    assert call.state.resume == {"have": ["salt", "pepper"]}
    # Pause and resume must share a thread_id or the answer joins a new
    # conversation.
    assert call.config == {"configurable": {"thread_id": session_id}}


async def test_a_resumed_turn_persists_no_duplicate_reply(
    authed_client: httpx.AsyncClient,
    fake_agent: FakeAgentGraph,
    with_address: SelectAddress,
) -> None:
    # Everything in a resumed run's `messages` is replayed history, so taking
    # the "last" reply from it would repost the recipe bubble from the turn
    # that paused.
    session_id = (await authed_client.post("/chat/sessions")).json()["id"]
    await with_address(session_id)
    fake_agent.interrupts = (FakeInterrupt(id="int-1", value={"ui": "checklist"}),)

    response = await authed_client.post(
        f"/chat/sessions/{session_id}/messages",
        json={"role": "user", "kind": "checklist", "payload": {"have": ["salt"]}},
    )

    assert response.json()["assistant_message"] is None
    transcript = (await authed_client.get(f"/chat/sessions/{session_id}")).json()
    assert [m["kind"] for m in transcript["messages"]] == ["checklist"]


#: A cart the way `report_cart` renders one (CUE-92).
CART_REPORT: dict[str, Any] = {
    "plan_id": 7,
    "summary": "Cart ready: 1 item, \u20b9180.00.",
    "below_minimum": False,
    "subtotal": "180.00",
    "minimum_order_value": "99.00",
    "shortfall": "0",
    "cart_total": "180.00",
    "items": [
        {
            "ingredient_name": "paneer",
            "status": "matched",
            "in_cart": True,
            "product_name": "Amul Paneer",
            "pack_size": "200 g",
            "quantity": 2,
            "unit_price": "90.00",
            "line_total": "180.00",
            "substitution_reason": None,
        }
    ],
}


async def test_a_turn_that_ends_in_a_cart_persists_it_as_cart_ready(
    authed_client: httpx.AsyncClient,
    fake_agent: FakeAgentGraph,
    with_address: SelectAddress,
) -> None:
    # A cart turn ends on its card, not on prose. `content` carries the summary
    # so a text-only client still says something useful; `payload` is the card.
    session_id = (await authed_client.post("/chat/sessions")).json()["id"]
    await with_address(session_id)
    fake_agent.cart_report = CART_REPORT

    response = await authed_client.post(
        f"/chat/sessions/{session_id}/messages",
        json={"role": "user", "content": "paneer butter masala"},
    )

    assert response.status_code == 201
    assistant = response.json()["assistant_message"]
    assert assistant["kind"] == "cart_ready"
    assert assistant["content"] == CART_REPORT["summary"]
    assert assistant["payload"]["items"][0]["ingredient_name"] == "paneer"


async def test_a_resumed_turn_ends_on_the_cart_it_produced(
    authed_client: httpx.AsyncClient,
    fake_agent: FakeAgentGraph,
    with_address: SelectAddress,
) -> None:
    # The resume is the turn that actually buys things, so unlike the
    # reply-less resume above it does owe the user a message.
    session_id = (await authed_client.post("/chat/sessions")).json()["id"]
    await with_address(session_id)
    fake_agent.interrupts = (FakeInterrupt(id="int-1", value={"ui": "checklist"}),)
    fake_agent.cart_report = CART_REPORT

    response = await authed_client.post(
        f"/chat/sessions/{session_id}/messages",
        json={"role": "user", "kind": "checklist", "payload": {"have": ["salt"]}},
    )

    assert response.json()["assistant_message"]["kind"] == "cart_ready"
    transcript = (await authed_client.get(f"/chat/sessions/{session_id}")).json()
    assert [m["kind"] for m in transcript["messages"]] == ["checklist", "cart_ready"]
