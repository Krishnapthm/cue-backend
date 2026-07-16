from __future__ import annotations

import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat import ChatSession
from app.models.user import User


@pytest_asyncio.fixture
async def chat_session(db_session: AsyncSession, linked_user: User) -> ChatSession:
    """A chat session owned by `linked_user`, to hang a `CartPlan` off of."""
    session_row = ChatSession(user_id=linked_user.id)
    db_session.add(session_row)
    await db_session.commit()
    await db_session.refresh(session_row)
    return session_row
