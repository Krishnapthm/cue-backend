"""`GET /chat/sessions/{id}/stream` and `/state`: the event sequence.

Runs the real router -> service -> graph path with the graph swapped at the
dependency, per AGENTS.md, over `httpx.AsyncClient` + `ASGITransport`. The
stub graph replays `(stream_mode, chunk)` pairs, so these tests assert on what
the client actually receives rather than on how LangGraph happens to chunk.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport
from langchain_core.messages import AIMessage, AIMessageChunk
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.exceptions import RecipeGenerationError
from app.chat.constants import ADDRESS_REQUIRED_MESSAGE
from app.chat.dependencies import agent_graph
from app.chat.schemas import RecoveryAction, StreamErrorCode
from app.database import get_session
from app.instamart.exceptions import InstamartAuthError, InstamartTransportError
from app.main import app
from app.models.chat import ChatSession
from app.models.user import User
from tests.chat.conftest import FakeAgentGraph, FakeInterrupt, SelectAddress


@pytest_asyncio.fixture
async def client(
    db_session: AsyncSession, fake_agent: FakeAgentGraph, user: User
) -> AsyncGenerator[httpx.AsyncClient]:
    from app.auth.dependencies import current_user

    async def override_get_session() -> AsyncGenerator[AsyncSession]:
        yield db_session

    async def override_agent_graph() -> FakeAgentGraph:
        return fake_agent

    async def override_current_user() -> User:
        return user

    app.dependency_overrides[get_session] = override_get_session
    app.dependency_overrides[agent_graph] = override_agent_graph
    app.dependency_overrides[current_user] = override_current_user
    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def session_id(
    client: httpx.AsyncClient, with_address: SelectAddress
) -> uuid.UUID:
    """A session ready to run turns: created, owned by `user`, addressed."""
    created = (await client.post("/chat/sessions")).json()["id"]
    await with_address(created)
    return uuid.UUID(created)


def _frames(body: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse an SSE body into `(event name, payload)` pairs."""
    frames = []
    for block in body.strip().split("\n\n"):
        if not block or block.startswith(":"):
            continue
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        frames.append((lines["event"], json.loads(lines["data"])))
    return frames


async def _stream(
    client: httpx.AsyncClient, session_id: uuid.UUID, message: str = "hi"
) -> httpx.Response:
    return await client.get(
        f"/chat/sessions/{session_id}/stream", params={"message": message}
    )


# --- the happy path --------------------------------------------------------


async def test_the_stream_is_an_event_stream(
    client: httpx.AsyncClient, session_id: uuid.UUID
) -> None:
    response = await _stream(client, session_id)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    # Proxies that buffer would defeat the point of streaming at all.
    assert response.headers["x-accel-buffering"] == "no"


async def test_a_turn_ends_with_exactly_one_done_carrying_the_reply(
    client: httpx.AsyncClient, session_id: uuid.UUID, fake_agent: FakeAgentGraph
) -> None:
    response = await _stream(client, session_id, "paneer butter masala")

    frames = _frames(response.text)
    names = [name for name, _ in frames]
    assert names.count("done") == 1
    assert names[-1] == "done"
    assert frames[-1][1]["reply"] == fake_agent.reply
    assert frames[-1][1]["interrupted"] is False


async def test_the_reply_is_persisted_to_the_transcript(
    client: httpx.AsyncClient, session_id: uuid.UUID, fake_agent: FakeAgentGraph
) -> None:
    await _stream(client, session_id, "paneer butter masala")

    transcript = (await client.get(f"/chat/sessions/{session_id}")).json()
    assert [(m["role"], m["content"]) for m in transcript["messages"]] == [
        ("user", "paneer butter masala"),
        ("assistant", fake_agent.reply),
    ]
    done = _frames((await _stream(client, session_id, "again")).text)[-1][1]
    assert isinstance(done["message_id"], int)


async def test_node_updates_arrive_as_stage_events(
    client: httpx.AsyncClient, session_id: uuid.UUID, fake_agent: FakeAgentGraph
) -> None:
    fake_agent.chunks = [
        ("updates", {"route_turn": {"turn_intent": "recipe"}}),
        ("updates", {"generate_recipe": {"messages": [AIMessage(content="done")]}}),
    ]

    frames = _frames((await _stream(client, session_id)).text)

    stages = [payload["node"] for name, payload in frames if name == "stage"]
    assert stages == ["route_turn", "generate_recipe"]


async def test_the_turn_runs_on_the_sessions_thread_with_a_context(
    client: httpx.AsyncClient, session_id: uuid.UUID, fake_agent: FakeAgentGraph
) -> None:
    await _stream(client, session_id)

    call = fake_agent.calls[0]
    assert call.config is not None
    assert call.config["configurable"]["thread_id"] == str(session_id)
    assert call.context is not None
    assert call.context.address_id == "addr-1"


# --- tokens ----------------------------------------------------------------


async def test_prose_tokens_stream_as_token_events(
    client: httpx.AsyncClient, session_id: uuid.UUID, fake_agent: FakeAgentGraph
) -> None:
    fake_agent.chunks = [
        (
            "messages",
            (AIMessageChunk(content="Your "), {"langgraph_node": "order_status"}),
        ),
        (
            "messages",
            (AIMessageChunk(content="order "), {"langgraph_node": "order_status"}),
        ),
        (
            "messages",
            (AIMessageChunk(content="is close."), {"langgraph_node": "order_status"}),
        ),
        (
            "updates",
            {"order_status": {"messages": [AIMessage(content="Your order is close.")]}},
        ),
    ]

    frames = _frames((await _stream(client, session_id, "where is my order")).text)

    tokens = [payload["text"] for name, payload in frames if name == "token"]
    assert tokens == ["Your ", "order ", "is close."]


async def test_structured_output_tokens_never_reach_the_client(
    client: httpx.AsyncClient, session_id: uuid.UUID, fake_agent: FakeAgentGraph
) -> None:
    # Every other model call in the graph runs under `with_structured_output`,
    # so its tokens are JSON fragments of internal schemas - including
    # `GuardrailDecision.reason`, which is attacker-influenceable text that
    # must never be rendered. The allowlist is what keeps it out.
    leak = '{"intent": "out_of_scope", "reason": "user asked for a python script"}'
    fake_agent.chunks = [
        ("messages", (AIMessageChunk(content=leak), {"langgraph_node": "route_turn"})),
        (
            "messages",
            (
                AIMessageChunk(content='{"dish_name": "x"}'),
                {"langgraph_node": "generate_recipe"},
            ),
        ),
        (
            "updates",
            {
                "refuse": {
                    "messages": [AIMessage(content="I can only help with cooking.")]
                }
            },
        ),
    ]

    response = await _stream(client, session_id, "write me a python script")

    assert "token" not in [name for name, _ in _frames(response.text)]
    assert "python script" not in response.text
    assert "reason" not in response.text


# --- matches ---------------------------------------------------------------


async def test_custom_payloads_arrive_as_keyed_match_events(
    client: httpx.AsyncClient, session_id: uuid.UUID, fake_agent: FakeAgentGraph
) -> None:
    # The workers finish out of order while the UI lists ingredients in recipe
    # order, so each event carries the ingredient name as a stable key and the
    # client fills that row in place.
    fake_agent.chunks = [
        ("custom", {"ingredient_name": "butter", "status": "matched", "spin_id": "s2"}),
        (
            "custom",
            {
                "ingredient_name": "paneer",
                "status": "substituted",
                "spin_id": "s1",
                "substitution_reason": "Amul 500g out of stock",
            },
        ),
        (
            "updates",
            {"report_cart": {"messages": [AIMessage(content="Basket ready.")]}},
        ),
    ]

    frames = _frames((await _stream(client, session_id)).text)

    matches = [payload for name, payload in frames if name == "match"]
    assert [m["ingredient_name"] for m in matches] == ["butter", "paneer"]
    assert matches[1]["status"] == "substituted"
    assert matches[1]["substitution_reason"] == "Amul 500g out of stock"


async def test_an_unrecognized_custom_payload_is_dropped_not_forwarded(
    client: httpx.AsyncClient, session_id: uuid.UUID, fake_agent: FakeAgentGraph
) -> None:
    # The client's contract is this app's event types, not whatever shape a
    # node happened to write.
    fake_agent.chunks = [
        ("custom", {"something": "unexpected"}),
        ("updates", {"refuse": {"messages": [AIMessage(content="no")]}}),
    ]

    response = await _stream(client, session_id)

    assert "match" not in [name for name, _ in _frames(response.text)]
    assert "unexpected" not in response.text


# --- interrupts ------------------------------------------------------------


async def test_an_interrupt_surfaces_as_its_own_event_with_its_payload(
    client: httpx.AsyncClient, session_id: uuid.UUID, fake_agent: FakeAgentGraph
) -> None:
    checklist = {"question": "Swap Amul for Nandini?", "items": ["paneer"]}
    fake_agent.chunks = [
        ("updates", {"compose_cart": {"cart_plan_id": 1}}),
        ("updates", {"__interrupt__": [FakeInterrupt(id="int-1", value=checklist)]}),
    ]

    frames = _frames((await _stream(client, session_id)).text)

    interrupts = [payload for name, payload in frames if name == "interrupt"]
    assert len(interrupts) == 1
    assert interrupts[0]["id"] == "int-1"
    assert interrupts[0]["payload"] == checklist
    assert frames[-1] == (
        "done",
        {"event": "done", "reply": None, "message_id": None, "interrupted": True},
    )


async def test_a_paused_turn_persists_no_assistant_reply(
    client: httpx.AsyncClient, session_id: uuid.UUID, fake_agent: FakeAgentGraph
) -> None:
    # The turn owes the user a decision, not an answer. A half-finished
    # assistant bubble would claim it was answered.
    fake_agent.chunks = [
        ("updates", {"compose_cart": {"messages": [AIMessage(content="partial")]}}),
        ("updates", {"__interrupt__": [FakeInterrupt(id="int-1", value={"q": "?"})]}),
    ]

    await _stream(client, session_id, "paneer butter masala")

    transcript = (await client.get(f"/chat/sessions/{session_id}")).json()
    assert [m["role"] for m in transcript["messages"]] == ["user"]


# --- errors mid-stream -----------------------------------------------------


async def test_auth_expiry_mid_stream_is_a_reconnect_event_not_a_500(
    client: httpx.AsyncClient, session_id: uuid.UUID, fake_agent: FakeAgentGraph
) -> None:
    # By the time this happens the response has already started, so the status
    # code is long gone.
    fake_agent.raises = InstamartAuthError()

    response = await _stream(client, session_id)

    assert response.status_code == 200
    frames = _frames(response.text)
    names = [name for name, _ in frames]
    assert names == ["error", "done"]
    assert frames[0][1]["code"] == StreamErrorCode.PROVIDER_AUTH.value
    assert frames[0][1]["action"] == RecoveryAction.RECONNECT_SWIGGY.value


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (InstamartTransportError(), StreamErrorCode.PROVIDER_UNAVAILABLE),
        (RecipeGenerationError(), StreamErrorCode.AGENT_FAILED),
    ],
    ids=["swiggy-unreachable", "agent-failed"],
)
async def test_other_domain_failures_close_the_stream_cleanly(
    client: httpx.AsyncClient,
    session_id: uuid.UUID,
    fake_agent: FakeAgentGraph,
    error: Exception,
    code: StreamErrorCode,
) -> None:
    fake_agent.raises = error

    frames = _frames((await _stream(client, session_id)).text)

    assert [name for name, _ in frames] == ["error", "done"]
    assert frames[0][1]["code"] == code.value
    assert frames[0][1]["action"] == RecoveryAction.RETRY.value


async def test_the_user_message_survives_a_mid_stream_failure(
    client: httpx.AsyncClient, session_id: uuid.UUID, fake_agent: FakeAgentGraph
) -> None:
    fake_agent.raises = InstamartAuthError()

    await _stream(client, session_id, "paneer butter masala")

    transcript = (await client.get(f"/chat/sessions/{session_id}")).json()
    assert [(m["role"], m["content"]) for m in transcript["messages"]] == [
        ("user", "paneer butter masala")
    ]


# --- preconditions ---------------------------------------------------------


async def test_streaming_another_users_session_404s_before_the_stream_starts(
    client: httpx.AsyncClient,
    db_session: AsyncSession,
    other_user: User,
    fake_agent: FakeAgentGraph,
) -> None:
    other_session = ChatSession(user_id=other_user.id)
    db_session.add(other_session)
    await db_session.commit()
    await db_session.refresh(other_session)

    response = await _stream(client, other_session.id)

    # Still a real status code: nothing has been written to the body yet.
    assert response.status_code == 404
    assert fake_agent.calls == []


async def test_streaming_an_unknown_session_404s(client: httpx.AsyncClient) -> None:
    response = await _stream(client, uuid.uuid4())

    assert response.status_code == 404


async def test_a_turn_without_an_address_streams_the_picker_prompt(
    client: httpx.AsyncClient, fake_agent: FakeAgentGraph
) -> None:
    unaddressed = (await client.post("/chat/sessions")).json()["id"]

    frames = _frames((await _stream(client, uuid.UUID(unaddressed))).text)

    assert [name for name, _ in frames] == ["done"]
    assert frames[0][1]["reply"] == ADDRESS_REQUIRED_MESSAGE
    assert fake_agent.calls == []


async def test_an_empty_message_is_rejected(
    client: httpx.AsyncClient, session_id: uuid.UUID
) -> None:
    response = await client.get(
        f"/chat/sessions/{session_id}/stream", params={"message": ""}
    )

    assert response.status_code == 422


# --- the state endpoint ----------------------------------------------------


async def test_state_reports_no_pending_interrupt_on_an_idle_session(
    client: httpx.AsyncClient, session_id: uuid.UUID
) -> None:
    response = await client.get(f"/chat/sessions/{session_id}/state")

    assert response.status_code == 200
    assert response.json() == {"pending_interrupt": None}


async def test_state_returns_the_pending_interrupt_after_a_dropped_connection(
    client: httpx.AsyncClient, session_id: uuid.UUID, fake_agent: FakeAgentGraph
) -> None:
    # The checkpointer already persisted it, so a client that lost the stream
    # (backgrounded app, cold start the next day) can still discover that a
    # decision is owed.
    checklist = {"question": "Swap Amul for Nandini?"}
    fake_agent.interrupts = (FakeInterrupt(id="int-1", value=checklist),)

    body = (await client.get(f"/chat/sessions/{session_id}/state")).json()

    assert body["pending_interrupt"] == {"id": "int-1", "payload": checklist}


async def test_state_404s_for_another_users_session(
    client: httpx.AsyncClient, db_session: AsyncSession, other_user: User
) -> None:
    other_session = ChatSession(user_id=other_user.id)
    db_session.add(other_session)
    await db_session.commit()
    await db_session.refresh(other_session)

    response = await client.get(f"/chat/sessions/{other_session.id}/state")

    assert response.status_code == 404
