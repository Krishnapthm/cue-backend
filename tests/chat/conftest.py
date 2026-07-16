from __future__ import annotations

import uuid

import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

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
