from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import CurrentUser
from app.chat import service
from app.chat.dependencies import AgentGraph
from app.chat.schemas import (
    CreateMessageRequest,
    MessageExchange,
    MessageKind,
    MessageResponse,
    MessageRole,
    SessionAgentState,
    SessionDetail,
    SessionSummary,
    UpdateSessionRequest,
)
from app.chat.sse import format_event
from app.database import DbSession
from app.models.user import User

#: Told to proxies that would otherwise buffer the response and defeat the
#: point of streaming it.
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

router = APIRouter(prefix="/chat/sessions", tags=["chat"])


@router.post(
    "",
    response_model=SessionSummary,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new, untitled chat session",
)
async def create(user: CurrentUser, session: DbSession) -> SessionSummary:
    """Start a new chat session for the signed-in Cue user (R8.1)."""
    chat_session = await service.create_session(session, user.id)
    return SessionSummary.model_validate(chat_session)


@router.get(
    "",
    response_model=list[SessionSummary],
    status_code=status.HTTP_200_OK,
    summary="List the caller's chat sessions, most recently updated first",
)
async def list_recents(user: CurrentUser, session: DbSession) -> list[SessionSummary]:
    """Return the caller's Recents list (R8.1); no date grouping or search."""
    sessions = await service.list_sessions(session, user.id)
    return [SessionSummary.model_validate(chat_session) for chat_session in sessions]


@router.patch(
    "/{session_id}",
    response_model=SessionSummary,
    status_code=status.HTTP_200_OK,
    summary="Set the delivery address for a chat session",
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Session not found"},
    },
)
async def update_session(
    session_id: uuid.UUID,
    request: UpdateSessionRequest,
    user: CurrentUser,
    session: DbSession,
) -> SessionSummary:
    """Save the caller-selected Swiggy delivery address for this session."""
    chat_session = await service.set_selected_address(
        session, user.id, session_id, request.selected_address_id
    )
    return SessionSummary.model_validate(chat_session)


@router.get(
    "/{session_id}",
    response_model=SessionDetail,
    status_code=status.HTTP_200_OK,
    summary="Get a chat session and its ordered message transcript",
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Session not found"},
    },
)
async def get(
    session_id: uuid.UUID, user: CurrentUser, session: DbSession
) -> SessionDetail:
    """Return a chat session and its transcript, scoped to the caller.

    Raises:
        ChatSessionNotFoundError: If `session_id` does not exist or belongs
            to another user (both surface as 404).
    """
    chat_session = await service.get_session(session, user.id, session_id)
    messages = await service.list_messages(session, session_id)
    return SessionDetail(
        id=chat_session.id,
        title=chat_session.title,
        selected_address_id=chat_session.selected_address_id,
        messages=[MessageResponse.model_validate(message) for message in messages],
    )


async def _sse_frames(
    session: AsyncSession,
    graph: AgentGraph,
    user: User,
    session_id: uuid.UUID,
    request: CreateMessageRequest,
) -> AsyncIterator[str]:
    """Encode a streamed turn's events as SSE frames."""
    async for event in service.stream_turn(
        session, graph, user.id, session_id, request
    ):
        yield format_event(event)


@router.get(
    "/{session_id}/stream",
    summary="Run a turn and stream its progress as server-sent events",
    response_class=StreamingResponse,
    responses={
        status.HTTP_200_OK: {
            "content": {"text/event-stream": {}},
            "description": (
                "A stream of named events: token, match, stage, interrupt, "
                "error, and a final done."
            ),
        },
        status.HTTP_404_NOT_FOUND: {"description": "Session not found"},
    },
)
async def stream(
    session_id: uuid.UUID,
    user: CurrentUser,
    session: DbSession,
    graph: AgentGraph,
    message: str = Query(min_length=1, description="The user's turn."),
) -> StreamingResponse:
    """Stream one user turn, event by event, as the agent works through it.

    A GET with the turn in the query string, because that is what `EventSource`
    can issue - the POST message endpoint stays for clients that want the whole
    turn in one payload.

    Ownership is checked before the response starts, so an unauthorized
    request still 404s properly. Once the first byte is out the status code is
    settled, and any later failure arrives as an `error` event instead.

    Raises:
        ChatSessionNotFoundError: If `session_id` does not exist or belongs
            to another user (both surface as 404, before any agent work).
    """
    await service.get_session(session, user.id, session_id)
    request = CreateMessageRequest(
        role=MessageRole.USER, kind=MessageKind.TEXT, content=message
    )
    return StreamingResponse(
        _sse_frames(session, graph, user, session_id, request),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.get(
    "/{session_id}/state",
    response_model=SessionAgentState,
    status_code=status.HTTP_200_OK,
    summary="Get any decision the agent is waiting on for this session",
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Session not found"},
    },
)
async def get_agent_state(
    session_id: uuid.UUID,
    user: CurrentUser,
    session: DbSession,
    graph: AgentGraph,
) -> SessionAgentState:
    """Return the session's pending interrupt, or null when it is idle.

    The stream drops whenever the app is backgrounded, so this is how a client
    rediscovers that a decision is owed - on reconnect, or on a cold start
    days later.

    Raises:
        ChatSessionNotFoundError: If `session_id` does not exist or belongs
            to another user (both surface as 404).
    """
    await service.get_session(session, user.id, session_id)
    return await service.pending_interrupt(graph, session_id)


@router.post(
    "/{session_id}/messages",
    response_model=MessageExchange,
    status_code=status.HTTP_201_CREATED,
    summary="Append a message to a chat session and run the agent on it",
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Session not found"},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "description": "Message body does not match its declared kind"
        },
        status.HTTP_502_BAD_GATEWAY: {
            "description": "The agent could not produce a reply"
        },
    },
)
async def add_message(
    session_id: uuid.UUID,
    request: CreateMessageRequest,
    user: CurrentUser,
    session: DbSession,
    graph: AgentGraph,
) -> MessageExchange:
    """Append a message, run the agent on it, and return both sides of the turn.

    Only a user's text turn reaches the agent; every other kind or role
    persists exactly as before and comes back with `assistant_message=None`.

    Raises:
        ChatSessionNotFoundError: If `session_id` does not exist or belongs
            to another user (both surface as 404, before any agent work).
        RecipeGenerationError: If the agent could not produce a recipe
            (surfaces as 502). The user's message stays persisted.
    """
    user_message, assistant_message = await service.run_turn(
        session, graph, user.id, session_id, request
    )
    return MessageExchange(
        user_message=MessageResponse.model_validate(user_message),
        assistant_message=(
            MessageResponse.model_validate(assistant_message)
            if assistant_message is not None
            else None
        ),
    )
