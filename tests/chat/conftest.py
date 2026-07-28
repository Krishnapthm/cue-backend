from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import pytest
import pytest_asyncio
from langchain_core.messages import AIMessage
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.context import CueContext
from app.models.chat import ChatSession
from app.models.user import User


@pytest_asyncio.fixture
async def user(db_session: AsyncSession) -> User:
    """A persisted Cue user to own chat sessions."""
    new_user = User(
        firebase_uid=f"firebase-uid-{uuid.uuid4()}", email="user@example.com"
    )
    db_session.add(new_user)
    await db_session.commit()
    await db_session.refresh(new_user)
    return new_user


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
async def chat_session(db_session: AsyncSession, user: User) -> ChatSession:
    """A persisted, untitled chat session owned by `user`."""
    session = ChatSession(user_id=user.id)
    db_session.add(session)
    await db_session.commit()
    await db_session.refresh(session)
    return session


@dataclass
class AgentCall:
    """One recorded invocation of the stub graph."""

    state: dict[str, Any]
    config: dict[str, Any] | None
    context: CueContext | None


@dataclass
class FakeAgentGraph:
    """Stands in for the compiled agent graph in `agent_graph`.

    Installed through `app.dependency_overrides`, not by monkeypatching the
    service, so the tests exercise the real router -> service -> graph path
    and can still assert that a turn made no agent call at all.
    """

    reply: str = "Here's what you'll need..."
    raises: Exception | None = None
    calls: list[AgentCall] = field(default_factory=list)

    async def ainvoke(
        self,
        state: dict[str, Any],
        config: dict[str, Any] | None = None,
        *,
        context: CueContext | None = None,
    ) -> dict[str, Any]:
        self.calls.append(AgentCall(state=state, config=config, context=context))
        if self.raises is not None:
            raise self.raises
        return {
            **state,
            "messages": [*state["messages"], AIMessage(content=self.reply)],
        }


#: Signature of the `with_address` fixture: (session_id, address_id=...) -> None.
SelectAddress = Callable[..., Awaitable[None]]


@pytest.fixture
def with_address(db_session: AsyncSession) -> SelectAddress:
    """Select a delivery address on a session, as the address picker does.

    A turn on a session with no address is answered without invoking the
    graph (Swiggy binds a cart to an address), so any test that expects the
    agent to run has to satisfy that precondition first.
    """

    async def _select(session_id: str | uuid.UUID, address_id: str = "addr-1") -> None:
        await db_session.execute(
            update(ChatSession)
            .where(ChatSession.id == uuid.UUID(str(session_id)))
            .values(selected_address_id=address_id)
        )
        await db_session.commit()

    return _select


@pytest.fixture
def fake_agent() -> FakeAgentGraph:
    """The stub graph every chat-router test runs against."""
    return FakeAgentGraph()
