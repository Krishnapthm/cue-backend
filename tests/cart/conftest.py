from __future__ import annotations

import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.cart import CartPlan
from app.models.chat import ChatSession
from app.models.user import User


@pytest_asyncio.fixture
async def chat_session(db_session: AsyncSession, user: User) -> ChatSession:
    """A chat session owned by `user`, to hang a `CartPlan` off of.

    Depends on the plain `user` fixture, not `linked_user`: a test asserting
    "not linked" behavior must be able to request `user` and `chat_session`
    together without `linked_user` ever being instantiated (fixtures are
    cached per test by name, so pulling in `linked_user` anywhere in the
    graph would silently link this same user).
    """
    session_row = ChatSession(user_id=user.id)
    db_session.add(session_row)
    await db_session.commit()
    await db_session.refresh(session_row)
    return session_row


@pytest_asyncio.fixture
async def cart_plan(db_session: AsyncSession, chat_session: ChatSession) -> CartPlan:
    """A live (non-superseded) `CartPlan` for `chat_session`, to check out."""
    plan = CartPlan(session_id=chat_session.id, address_id="addr-1")
    db_session.add(plan)
    await db_session.commit()
    await db_session.refresh(plan)
    return plan
