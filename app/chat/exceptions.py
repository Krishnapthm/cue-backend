from __future__ import annotations

from fastapi import status

from app.exceptions import AppError, NotFoundError


class ChatSessionNotFoundError(NotFoundError):
    """The requested chat session does not exist or is not owned by the caller."""

    detail = "Chat session not found."


class NoPendingChecklistError(AppError):
    """A checklist answer arrived for a session that is not waiting on one.

    Raised rather than quietly accepted, because the alternative is worse than an
    error: a `Command(resume=...)` on a thread with no pending interrupt does not
    fail - it starts a fresh run, and the session looks stuck for reasons nothing
    in the logs explains. Failing loudly here is what makes that impossible.
    """

    status_code = status.HTTP_409_CONFLICT
    detail = "This chat session is not waiting on a checklist answer."


class InvalidChecklistAnswerError(AppError):
    """A structured interrupt answer did not match its pending card.

    A resume we cannot read is not consent. Defaulting it to "none of them"
    would silently buy the user everything they already own.
    """

    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    detail = "This answer does not match the pending choice."
