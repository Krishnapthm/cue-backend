from __future__ import annotations

from fastapi import APIRouter, status

from app.auth.dependencies import CurrentUser
from app.auth.schemas import UserResponse
from app.models.user import User

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get(
    "/me",
    response_model=UserResponse,
    status_code=status.HTTP_200_OK,
    summary="Return the signed-in Cue user",
    responses={
        status.HTTP_401_UNAUTHORIZED: {
            "description": "Missing, expired, or malformed Firebase ID token"
        },
    },
)
async def me(user: CurrentUser) -> User:
    """Return the caller's Cue user row.

    A cheap gate the client can run right after Firebase sign-in to answer
    "is my token good and does my Cue user exist" without firing a chat
    mutation to find out - and a clean place to detect a stale token.

    Provisioning is not done here: the `current_user` dependency already
    upserts the row on first sign-in, so this endpoint only makes that
    existing path easy to trigger and observe.

    The raw ORM row is returned and `response_model` validates it, rather
    than validating twice (see AGENTS.md).
    """
    return user
