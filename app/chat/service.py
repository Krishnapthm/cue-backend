"""Chat session and message persistence (R8.1/R8.3, CUE-20).

This is the display transcript only - the LangGraph checkpointer (CUE-21)
owns the agent's own state under the same `chat_session.id` as its
`thread_id`. The two are deliberately decoupled; see `app/models/chat.py`.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import ValidationError
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.context import CueContext
from app.agent.exceptions import RecipeGenerationError
from app.agent.graph import PROSE_NODES, CueGraph, thread_config
from app.agent.schemas import MatchResult
from app.agent.state import AgentState
from app.chat.constants import ADDRESS_REQUIRED_MESSAGE
from app.chat.exceptions import ChatSessionNotFoundError
from app.chat.schemas import (
    ChatStreamEvent,
    CreateMessageRequest,
    DoneEvent,
    ErrorEvent,
    InterruptEvent,
    MatchEvent,
    MessageKind,
    MessageRole,
    PendingInterrupt,
    RecoveryAction,
    SessionAgentState,
    StageEvent,
    StreamErrorCode,
    TokenEvent,
)
from app.exceptions import AppError
from app.instamart.exceptions import (
    InstamartAuthError,
    InstamartDomainError,
    InstamartTransportError,
)
from app.models.chat import ChatMessage, ChatSession

logger = logging.getLogger(__name__)

#: The key LangGraph reports a pause under in `updates` mode. Spelled out
#: rather than imported: `langgraph.constants.INTERRUPT` is private as of
#: LangGraph v1 and slated for removal, and this is a stable wire-level name.
INTERRUPT_KEY = "__interrupt__"


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


def _turn_state(
    user_id: int, session_id: uuid.UUID, request: CreateMessageRequest
) -> AgentState:
    """Build the state literal one turn starts from.

    Only the user's new turn goes in: the rest of the transcript is replayed
    by the checkpointer under the same `thread_id`.
    """
    return {
        "session_id": str(session_id),
        "user_id": user_id,
        "messages": [HumanMessage(content=request.content or "")],
    }


async def _turn_context(
    session: AsyncSession, user_id: int, session_id: uuid.UUID
) -> CueContext | None:
    """Build the runtime context for one turn, if the session can run one.

    The context is built fresh per invocation and never checkpointed: it
    carries this request's `AsyncSession`, so a node's writes land in the same
    unit of work as the request that triggered them. See
    `app/agent/context.py`.

    Args:
        session: An active database session.
        user_id: The Cue user who must own `session_id`.
        session_id: The session being run.

    Returns:
        The context, or `None` when the session has no delivery address
        selected yet - a precondition of running a turn at all, since Swiggy
        binds a cart to an address.

    Raises:
        ChatSessionNotFoundError: If `session_id` does not exist or is not
            owned by `user_id`.
    """
    chat_session = await get_session(session, user_id, session_id)
    if not chat_session.selected_address_id:
        return None
    return CueContext(
        session=session,
        user_id=user_id,
        chat_session_id=session_id,
        address_id=chat_session.selected_address_id,
    )


async def _persist_reply(
    session: AsyncSession, user_id: int, session_id: uuid.UUID, reply: str
) -> ChatMessage:
    """Append one assistant text message to a session's transcript."""
    return await append_message(
        session,
        user_id,
        session_id,
        CreateMessageRequest(
            role=MessageRole.ASSISTANT, kind=MessageKind.TEXT, content=reply
        ),
    )


async def run_turn(
    session: AsyncSession,
    graph: CueGraph,
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

    A turn on a session with no delivery address selected is answered here,
    without invoking the graph: Swiggy binds a cart to an address, so the turn
    could not finish, and spending a model call on it would be waste. Address
    selection is a precondition, never something the agent decides.

    The whole turn is one blocking request. `stream_turn` is the streaming
    equivalent and is what the app uses; this stays for clients that want one
    payload, and for turns short enough not to need progress.

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

    context = await _turn_context(session, user_id, session_id)
    if context is None:
        prompt = await _persist_reply(
            session, user_id, session_id, ADDRESS_REQUIRED_MESSAGE
        )
        return user_message, prompt

    result = await graph.ainvoke(
        _turn_state(user_id, session_id, request),
        thread_config(str(session_id)),
        context=context,
    )

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

    assistant_message = await _persist_reply(session, user_id, session_id, reply)
    return user_message, assistant_message


#: Which domain failures become which `error` event. Anything not listed is a
#: bug rather than a condition the user can act on, and is left to raise.
_ERROR_EVENTS: dict[type[AppError], tuple[StreamErrorCode, RecoveryAction]] = {
    InstamartAuthError: (
        StreamErrorCode.PROVIDER_AUTH,
        RecoveryAction.RECONNECT_SWIGGY,
    ),
    InstamartTransportError: (
        StreamErrorCode.PROVIDER_UNAVAILABLE,
        RecoveryAction.RETRY,
    ),
    InstamartDomainError: (StreamErrorCode.PROVIDER_REJECTED, RecoveryAction.RETRY),
    RecipeGenerationError: (StreamErrorCode.AGENT_FAILED, RecoveryAction.RETRY),
}


def _error_event(exc: AppError) -> ErrorEvent | None:
    """Translate a domain error into the event that names its recovery."""
    for error_type, (code, action) in _ERROR_EVENTS.items():
        if isinstance(exc, error_type):
            return ErrorEvent(code=code, message=exc.detail, action=action)
    return None


def _token_event(chunk: Any) -> TokenEvent | None:
    """Build a token event from a `messages`-mode chunk, if it carries prose.

    LangGraph yields `(message_chunk, metadata)` here for *every* model call in
    the graph, including the structured-output ones whose tokens are JSON
    fragments of internal schemas. Only nodes on the `PROSE_NODES` allowlist
    are forwarded; see the note there on why it is an allowlist.
    """
    message, metadata = chunk
    if metadata.get("langgraph_node") not in PROSE_NODES:
        return None
    text = str(message.content)
    return TokenEvent(text=text) if text else None


def _custom_event(chunk: Any) -> MatchEvent | None:
    """Build a match event from a node's `get_stream_writer()` payload.

    Nodes write `MatchResult`-shaped payloads. Anything else is dropped rather
    than forwarded: the client's contract is this module's event types, not
    whatever a node happened to write.
    """
    try:
        return MatchEvent.from_match(MatchResult.model_validate(chunk))
    except ValidationError:
        logger.warning("Dropping unrecognized custom stream payload from a node.")
        return None


def _updates_events(chunk: dict[str, Any]) -> Iterator[StageEvent | InterruptEvent]:
    """Turn one `updates`-mode chunk into stage and interrupt events."""
    for node, node_update in chunk.items():
        if node == INTERRUPT_KEY:
            for interrupt in node_update:
                yield InterruptEvent(
                    id=getattr(interrupt, "id", None),
                    payload=getattr(interrupt, "value", None),
                )
            continue
        yield StageEvent(node=node)


def _messages_in(update: Any) -> list[BaseMessage]:
    """Return the messages a node's partial state update appended, if any."""
    if not isinstance(update, dict):
        return []
    messages = update.get("messages")
    return messages if isinstance(messages, list) else []


async def stream_turn(
    session: AsyncSession,
    graph: CueGraph,
    user_id: int,
    session_id: uuid.UUID,
    request: CreateMessageRequest,
) -> AsyncIterator[ChatStreamEvent]:
    """Run one turn, emitting typed events as the work happens.

    The blocking `run_turn` returns a single payload after several seconds of
    Swiggy calls and shows nothing in between. The chat design resolves
    ingredient rows one at a time as they come back, and the ingredient
    fan-out runs N parallel searches, so the designed screen cannot be built
    without this.

    Three LangGraph stream modes drive it: `messages` for prose tokens,
    `custom` for the per-ingredient payloads nodes write, and `updates` for
    stage changes and the `__interrupt__` pause. Every chunk is translated
    into one of this module's event types - raw LangGraph shapes are an
    internal contract that changes between versions and never reach a client.

    A domain failure after the first byte cannot be a status code: the
    response has already started. Those are emitted as an `error` event
    naming the action that recovers them, and the stream then closes cleanly.
    Anything not in `_ERROR_EVENTS` is a bug, and is left to raise.

    Args:
        session: An active database session.
        graph: The compiled agent graph for this request.
        user_id: The Cue user who must own `session_id`.
        session_id: The session to append to and run the agent against.
        request: The inbound message.

    Yields:
        The turn's events, always ending with exactly one `done`.

    Raises:
        ChatSessionNotFoundError: If `session_id` does not exist or is not
            owned by `user_id`. Raised before anything is yielded, so the
            caller can still answer with a 404.
    """
    await append_message(session, user_id, session_id, request)
    if not _runs_the_agent(request):
        yield DoneEvent()
        return

    context = await _turn_context(session, user_id, session_id)
    if context is None:
        prompt = await _persist_reply(
            session, user_id, session_id, ADDRESS_REQUIRED_MESSAGE
        )
        yield DoneEvent(reply=ADDRESS_REQUIRED_MESSAGE, message_id=prompt.id)
        return

    replies: list[BaseMessage] = []
    interrupted = False

    try:
        async for stream_mode, chunk in graph.astream(
            _turn_state(user_id, session_id, request),
            thread_config(str(session_id)),
            context=context,
            stream_mode=["messages", "custom", "updates"],
        ):
            match stream_mode:
                case "messages":
                    if (token := _token_event(chunk)) is not None:
                        yield token
                case "custom":
                    if (match := _custom_event(chunk)) is not None:
                        yield match
                case "updates":
                    updates = cast("dict[str, Any]", chunk)
                    for node_update in updates.values():
                        replies.extend(_messages_in(node_update))
                    for update_event in _updates_events(updates):
                        interrupted = interrupted or isinstance(
                            update_event, InterruptEvent
                        )
                        yield update_event
    except AppError as exc:
        error_event = _error_event(exc)
        if error_event is None:
            raise
        logger.warning(
            "Turn for session %s failed mid-stream: %s", session_id, exc.detail
        )
        yield error_event
        yield DoneEvent()
        return

    if interrupted:
        # A paused turn owes the user a decision, not a reply. Persisting a
        # half-finished assistant bubble here would leave the transcript
        # claiming the turn was answered.
        yield DoneEvent(interrupted=True)
        return

    reply = _reply_text(replies)
    if reply is None:
        logger.error(
            "Agent produced no reply for session %s; persisting the user message only.",
            session_id,
        )
        yield DoneEvent()
        return

    assistant_message = await _persist_reply(session, user_id, session_id, reply)
    yield DoneEvent(reply=reply, message_id=assistant_message.id)


async def pending_interrupt(
    graph: CueGraph, session_id: uuid.UUID
) -> SessionAgentState:
    """Return the interrupt a session is waiting on, if any.

    Read straight off the checkpointer, which has already persisted it, so
    this answers both a dropped SSE connection and a cold start the next day.

    Args:
        graph: The compiled agent graph for this request.
        session_id: The session to inspect; its `str()` is the `thread_id`.

    Returns:
        The pending interrupt, or an empty state when the session is idle.
    """
    snapshot = await graph.aget_state(thread_config(str(session_id)))
    interrupts = snapshot.interrupts
    if not interrupts:
        return SessionAgentState()
    first = interrupts[0]
    return SessionAgentState(
        pending_interrupt=PendingInterrupt(
            id=getattr(first, "id", None), payload=getattr(first, "value", None)
        )
    )
