"""Chat session and message persistence (R8.1/R8.3, CUE-20).

This is the display transcript only - the LangGraph checkpointer (CUE-21)
owns the agent's own state under the same `chat_session.id` as its
`thread_id`. The two are deliberately decoupled; see `app/models/chat.py`.
"""

from __future__ import annotations

import logging
import uuid

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.graph import thread_config
from app.agent.state import AgentState
from app.chat.exceptions import ChatSessionNotFoundError
from app.chat.schemas import (
    CreateMessageRequest,
    MessageKind,
    MessageRole,
)
from app.models.chat import ChatMessage, ChatSession

logger = logging.getLogger(__name__)


async def create_session(session: AsyncSession, user_id: int) -> ChatSession:
    """Create a new, untitled chat session for `user_id`.

    Args:
        session: An active database session.
        user_id: The owning Cue user.

    Returns:
        The newly created session.
    """
    chat_session = ChatSession(user_id=user_id)
    session.add(chat_session)
    await session.commit()
    await session.refresh(chat_session)
    return chat_session


async def list_sessions(session: AsyncSession, user_id: int) -> list[ChatSession]:
    """Return `user_id`'s chat sessions for the Recents list (R8.1).

    Most-recently-updated first, via `ix_chat_session_user_recents`. No date
    grouping or search - that is out of scope for CUE-20.

    Args:
        session: An active database session.
        user_id: The owning Cue user.

    Returns:
        The user's sessions, ordered by `updated_at` descending.
    """
    stmt = (
        select(ChatSession)
        .where(ChatSession.user_id == user_id)
        .order_by(ChatSession.updated_at.desc())
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def get_session(
    session: AsyncSession, user_id: int, session_id: uuid.UUID
) -> ChatSession:
    """Return a chat session, scoped to its owner.

    Args:
        session: An active database session.
        user_id: The Cue user who must own `session_id`.
        session_id: The session to fetch.

    Returns:
        The matching session.

    Raises:
        ChatSessionNotFoundError: If `session_id` does not exist, or exists
            but belongs to a different user - both cases return 404, never
            leaking another user's data.
    """
    stmt = select(ChatSession).where(
        ChatSession.id == session_id, ChatSession.user_id == user_id
    )
    result = await session.execute(stmt)
    chat_session = result.scalar_one_or_none()
    if chat_session is None:
        raise ChatSessionNotFoundError
    return chat_session


async def list_messages(
    session: AsyncSession, session_id: uuid.UUID
) -> list[ChatMessage]:
    """Return a session's transcript in display order.

    Ordered by `id`, not `created_at`, per `ix_chat_message_session` - `id`
    is monotonic and collision-free, `created_at` is not guaranteed to be.
    Does not check ownership; callers must authorize `session_id` first via
    `get_session`.

    Args:
        session: An active database session.
        session_id: The session whose messages to fetch.

    Returns:
        The session's messages, oldest first.
    """
    stmt = (
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.id)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def append_message(
    session: AsyncSession,
    user_id: int,
    session_id: uuid.UUID,
    request: CreateMessageRequest,
) -> ChatMessage:
    """Append a message to a session's transcript and resurface it in Recents.

    Bumps `chat_session.updated_at` so the session sorts to the top of
    Recents (R8.1) via the shared `trg_chat_session_updated_at` trigger,
    which stamps `now()` on any UPDATE regardless of the value written here.

    Args:
        session: An active database session.
        user_id: The Cue user who must own `session_id`.
        session_id: The session to append to.
        request: The message to append; already validated against the
            `ck_chat_message_body` shape by `CreateMessageRequest`.

    Returns:
        The newly persisted message.

    Raises:
        ChatSessionNotFoundError: If `session_id` does not exist or is not
            owned by `user_id`.
    """
    await get_session(session, user_id, session_id)

    message = ChatMessage(
        session_id=session_id,
        role=request.role.value,
        kind=request.kind.value,
        content=request.content,
        payload=request.payload,
    )
    session.add(message)

    bump_stmt = (
        update(ChatSession)
        .where(ChatSession.id == session_id)
        .values(updated_at=func.now())
    )
    await session.execute(bump_stmt)

    await session.commit()
    await session.refresh(message)
    return message


def _runs_the_agent(request: CreateMessageRequest) -> bool:
    """Return whether this message should be handed to the agent.

    Only a user's text turn does. Every other combination persists exactly
    as it always has and reports no assistant reply - preserving the
    behaviour the app's other write paths depend on, and making sure a
    checklist or image append never burns a model call.
    """
    return request.role is MessageRole.USER and request.kind is MessageKind.TEXT


def _reply_text(messages: list[BaseMessage]) -> str | None:
    """Return the content of the last `AIMessage` in a graph result."""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            content = str(message.content)
            return content or None
    return None


async def run_turn(
    session: AsyncSession,
    graph: CompiledStateGraph[AgentState],
    user_id: int,
    session_id: uuid.UUID,
    request: CreateMessageRequest,
) -> tuple[ChatMessage, ChatMessage | None]:
    """Persist an inbound message, run the agent on it, and persist the reply.

    The user's message is persisted first, and stays persisted even if the
    agent then fails: the user did send it, and rolling it back would lose
    input they typed. Ownership is checked before any of that (inside
    `append_message`), so an unauthorized request 404s without ever
    reaching the agent - and never burns a model call.

    The agent runs against `thread_id = str(session_id)`, so two sessions
    belonging to the same user keep entirely separate agent memory. That
    thread is the LangGraph checkpointer's own state, distinct from this
    display transcript; see the module docstring.

    The whole turn is one blocking request for now. Token streaming
    (`stream_mode="messages"`) is a separate issue - clients should not be
    built assuming this endpoint stays synchronous forever.

    Args:
        session: An active database session.
        graph: The compiled agent graph for this request.
        user_id: The Cue user who must own `session_id`.
        session_id: The session to append to and run the agent against.
        request: The inbound message.

    Returns:
        The persisted user message, and the persisted assistant reply - or
        `None` when the turn ran no agent, or the agent produced no reply.

    Raises:
        ChatSessionNotFoundError: If `session_id` does not exist or is not
            owned by `user_id`.
        RecipeGenerationError: If the agent could not produce a recipe. The
            user's message stays persisted; the global handler maps this to
            502.
    """
    user_message = await append_message(session, user_id, session_id, request)
    if not _runs_the_agent(request):
        return user_message, None

    state: AgentState = {
        "session_id": str(session_id),
        "user_id": user_id,
        "messages": [HumanMessage(content=request.content or "")],
    }
    result = await graph.ainvoke(state, thread_config(str(session_id)))

    reply = _reply_text(result["messages"])
    if reply is None:
        # Every branch of the graph emits a reply, so this means the graph
        # changed shape without this call site being updated. Persisting
        # nothing is better than persisting an empty assistant bubble.
        logger.error(
            "Agent produced no reply for session %s; persisting the user message only.",
            session_id,
        )
        return user_message, None

    assistant_message = await append_message(
        session,
        user_id,
        session_id,
        CreateMessageRequest(
            role=MessageRole.ASSISTANT, kind=MessageKind.TEXT, content=reply
        ),
    )
    return user_message, assistant_message
