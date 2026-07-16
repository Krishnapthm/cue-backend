from __future__ import annotations

import asyncio
import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat import service
from app.chat.exceptions import ChatSessionNotFoundError
from app.chat.schemas import CreateMessageRequest, MessageKind, MessageRole
from app.models.chat import ChatSession
from app.models.user import User


async def test_create_session_persists_an_untitled_session(
    db_session: AsyncSession, user: User
) -> None:
    session = await service.create_session(db_session, user.id)

    assert session.user_id == user.id
    assert session.title is None
    assert session.selected_address_id is None


async def test_list_sessions_returns_only_the_owning_users_sessions(
    db_session: AsyncSession, user: User, other_user: User
) -> None:
    mine = await service.create_session(db_session, user.id)
    await service.create_session(db_session, other_user.id)

    sessions = await service.list_sessions(db_session, user.id)

    assert [s.id for s in sessions] == [mine.id]


async def test_list_sessions_orders_most_recently_updated_first(
    db_session: AsyncSession, user: User
) -> None:
    first = await service.create_session(db_session, user.id)
    await asyncio.sleep(0.01)
    second = await service.create_session(db_session, user.id)

    sessions = await service.list_sessions(db_session, user.id)

    assert [s.id for s in sessions] == [second.id, first.id]


async def test_list_sessions_resurfaces_a_session_after_a_new_message(
    db_session: AsyncSession, user: User
) -> None:
    older = await service.create_session(db_session, user.id)
    await asyncio.sleep(0.01)
    newer = await service.create_session(db_session, user.id)
    assert [s.id for s in await service.list_sessions(db_session, user.id)] == [
        newer.id,
        older.id,
    ]

    await service.append_message(
        db_session,
        user.id,
        older.id,
        CreateMessageRequest(role=MessageRole.USER, content="hi"),
    )

    sessions = await service.list_sessions(db_session, user.id)
    assert [s.id for s in sessions] == [older.id, newer.id]


async def test_get_session_returns_the_owning_users_session(
    db_session: AsyncSession, user: User, chat_session: ChatSession
) -> None:
    fetched = await service.get_session(db_session, user.id, chat_session.id)

    assert fetched.id == chat_session.id


async def test_get_session_raises_for_an_unknown_session_id(
    db_session: AsyncSession, user: User
) -> None:
    with pytest.raises(ChatSessionNotFoundError):
        await service.get_session(db_session, user.id, uuid.uuid4())


async def test_get_session_raises_when_owned_by_another_user(
    db_session: AsyncSession, other_user: User, chat_session: ChatSession
) -> None:
    with pytest.raises(ChatSessionNotFoundError):
        await service.get_session(db_session, other_user.id, chat_session.id)


async def test_list_messages_is_empty_for_a_new_session(
    db_session: AsyncSession, chat_session: ChatSession
) -> None:
    messages = await service.list_messages(db_session, chat_session.id)

    assert messages == []


async def test_append_message_persists_a_text_message(
    db_session: AsyncSession, user: User, chat_session: ChatSession
) -> None:
    message = await service.append_message(
        db_session,
        user.id,
        chat_session.id,
        CreateMessageRequest(role=MessageRole.USER, content="What's for dinner?"),
    )

    assert message.session_id == chat_session.id
    assert message.role == "user"
    assert message.kind == "text"
    assert message.content == "What's for dinner?"
    assert message.payload is None


async def test_append_message_persists_a_non_text_message_with_payload(
    db_session: AsyncSession, user: User, chat_session: ChatSession
) -> None:
    message = await service.append_message(
        db_session,
        user.id,
        chat_session.id,
        CreateMessageRequest(
            role=MessageRole.ASSISTANT,
            kind=MessageKind.CHECKLIST,
            payload={"items": ["milk", "eggs"]},
        ),
    )

    assert message.kind == "checklist"
    assert message.content is None
    assert message.payload == {"items": ["milk", "eggs"]}


async def test_append_message_orders_messages_by_id(
    db_session: AsyncSession, user: User, chat_session: ChatSession
) -> None:
    first = await service.append_message(
        db_session,
        user.id,
        chat_session.id,
        CreateMessageRequest(role=MessageRole.USER, content="one"),
    )
    second = await service.append_message(
        db_session,
        user.id,
        chat_session.id,
        CreateMessageRequest(role=MessageRole.ASSISTANT, content="two"),
    )

    messages = await service.list_messages(db_session, chat_session.id)

    assert [m.id for m in messages] == [first.id, second.id]


async def test_append_message_bumps_the_sessions_updated_at(
    db_session: AsyncSession, user: User, chat_session: ChatSession
) -> None:
    original_updated_at = chat_session.updated_at

    await asyncio.sleep(0.01)
    await service.append_message(
        db_session,
        user.id,
        chat_session.id,
        CreateMessageRequest(role=MessageRole.USER, content="hi"),
    )

    refreshed = await service.get_session(db_session, user.id, chat_session.id)
    assert refreshed.updated_at > original_updated_at


async def test_append_message_raises_for_an_unknown_session_id(
    db_session: AsyncSession, user: User
) -> None:
    with pytest.raises(ChatSessionNotFoundError):
        await service.append_message(
            db_session,
            user.id,
            uuid.uuid4(),
            CreateMessageRequest(role=MessageRole.USER, content="hi"),
        )


async def test_append_message_raises_when_owned_by_another_user(
    db_session: AsyncSession, other_user: User, chat_session: ChatSession
) -> None:
    with pytest.raises(ChatSessionNotFoundError):
        await service.append_message(
            db_session,
            other_user.id,
            chat_session.id,
            CreateMessageRequest(role=MessageRole.USER, content="hi"),
        )
