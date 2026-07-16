from __future__ import annotations

from app.exceptions import NotFoundError


class ChatSessionNotFoundError(NotFoundError):
    """The requested chat session does not exist or is not owned by the caller."""

    detail = "Chat session not found."
