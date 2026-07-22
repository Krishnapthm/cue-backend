from __future__ import annotations

import uuid

from fastapi import APIRouter, status

from app.auth.dependencies import CurrentUser
from app.chat import service
from app.chat.dependencies import AgentGraph
from app.chat.schemas import (
    CreateMessageRequest,
    MessageExchange,
    MessageResponse,
    SessionDetail,
    SessionSummary,
)
from app.database import DbSession

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
