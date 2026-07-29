"""Tests for best-effort, once-only automatic chat-session titling."""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any, cast

import pytest
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.agent.config import ModelRole
from app.agent.context import CueContext
from app.agent.nodes import title as title_node
from app.agent.schemas import GeneratedRecipe
from app.agent.state import AgentState
from app.models.chat import ChatSession
from app.models.user import User


class _FakeTitleModel:
    def __init__(self, result: str | Exception) -> None:
        self.result = result
        self.roles: list[ModelRole] = []

    async def ainvoke(self, _prompt: list[Any]) -> AIMessage:
        if isinstance(self.result, Exception):
            raise self.result
        return AIMessage(content=self.result)


def _factory_for(db_session: AsyncSession) -> async_sessionmaker[AsyncSession]:
    bind = db_session.bind
    assert isinstance(bind, AsyncEngine)
    return async_sessionmaker(bind=bind, expire_on_commit=False, autoflush=False)


async def _session_title(db_session: AsyncSession, session_id: uuid.UUID) -> str | None:
    return await db_session.scalar(
        select(ChatSession.title).where(ChatSession.id == session_id)
    )


async def test_generate_title_persists_a_short_dish_name_once(
    monkeypatch: pytest.MonkeyPatch,
    db_session: AsyncSession,
    user: User,
) -> None:
    chat_session = ChatSession(user_id=user.id)
    db_session.add(chat_session)
    await db_session.commit()
    await db_session.refresh(chat_session)
    model = _FakeTitleModel("Chole masala")

    monkeypatch.setattr(title_node, "session_factory", _factory_for(db_session))

    def _model(role: ModelRole) -> _FakeTitleModel:
        model.roles.append(role)
        return model

    monkeypatch.setattr(title_node, "get_chat_model", _model)

    await title_node.generate_title(chat_session.id, "chole masala for dinner")
    await title_node.generate_title(chat_session.id, "different dish")

    assert await _session_title(db_session, chat_session.id) == "Chole masala"
    assert model.roles == [ModelRole.TITLE]


async def test_generate_title_failure_leaves_session_untitled(
    monkeypatch: pytest.MonkeyPatch,
    db_session: AsyncSession,
    user: User,
) -> None:
    chat_session = ChatSession(user_id=user.id)
    db_session.add(chat_session)
    await db_session.commit()
    await db_session.refresh(chat_session)

    monkeypatch.setattr(title_node, "session_factory", _factory_for(db_session))
    monkeypatch.setattr(
        title_node,
        "get_chat_model",
        lambda _role: _FakeTitleModel(RuntimeError("down")),
    )

    await title_node.generate_title(chat_session.id, "palak paneer")

    assert await _session_title(db_session, chat_session.id) is None


async def test_generate_title_truncates_to_database_limit(
    monkeypatch: pytest.MonkeyPatch,
    db_session: AsyncSession,
    user: User,
) -> None:
    chat_session = ChatSession(user_id=user.id)
    db_session.add(chat_session)
    await db_session.commit()
    await db_session.refresh(chat_session)
    overlong_title = "x" * 250

    monkeypatch.setattr(title_node, "session_factory", _factory_for(db_session))
    monkeypatch.setattr(
        title_node, "get_chat_model", lambda _role: _FakeTitleModel(overlong_title)
    )

    await title_node.generate_title(chat_session.id, "dal")

    title = await _session_title(db_session, chat_session.id)
    assert title == "x" * 200


async def test_schedule_title_node_never_blocks_later_turns(
    monkeypatch: pytest.MonkeyPatch, db_session: AsyncSession
) -> None:
    session_id = uuid.uuid4()
    scheduled: list[tuple[uuid.UUID, str]] = []
    monkeypatch.setattr(
        title_node,
        "_schedule_title_generation",
        lambda identifier, dish_name: scheduled.append((identifier, dish_name)),
    )
    state: AgentState = {
        "session_id": str(session_id),
        "user_id": 1,
        "messages": [],
        "recipe": GeneratedRecipe(
            dish_name="Palak paneer",
            estimated_time_minutes=30,
            ingredients=[],
            method_summary="Cook.",
        ),
    }
    runtime = SimpleNamespace(
        context=CueContext(
            session=db_session,
            user_id=1,
            chat_session_id=session_id,
            address_id="addr-1",
        )
    )

    update = await title_node.schedule_title_node(
        state, cast(Runtime[CueContext], runtime)
    )

    assert update == {"title_attempted": True}
    assert scheduled == [(session_id, "Palak paneer")]
    assert (
        await title_node.schedule_title_node(
            {**state, "title_attempted": True}, cast(Runtime[CueContext], runtime)
        )
        == {}
    )
