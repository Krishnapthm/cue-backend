from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, model_validator


class MessageKind(StrEnum):
    """The shape of a message body, mirrored from `ck_chat_message_kind_allowed`."""

    TEXT = "text"
    IMAGE = "image"
    CHECKLIST = "checklist"
    CART_READY = "cart_ready"


class MessageRole(StrEnum):
    """Who authored a chat message, mirrored from `ck_chat_message_role_allowed`."""

    USER = "user"
    ASSISTANT = "assistant"


class CreateMessageRequest(BaseModel):
    """A message to append to a chat session's transcript."""

    role: MessageRole
    kind: MessageKind = MessageKind.TEXT
    content: str | None = None
    payload: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _validate_body(self) -> Self:
        """Mirror the `ck_chat_message_body` DB constraint client-side.

        `kind="text"` requires `content`; every other kind requires `payload`.
        Enforced here so a violation returns 422 before it ever reaches the
        database constraint.
        """
        if self.kind is MessageKind.TEXT and self.content is None:
            raise ValueError("content is required when kind is 'text'.")
        if self.kind is not MessageKind.TEXT and self.payload is None:
            raise ValueError("payload is required when kind is not 'text'.")
        return self


class MessageResponse(BaseModel):
    """A persisted chat message, as returned in a transcript."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    role: MessageRole
    kind: MessageKind
    content: str | None
    payload: dict[str, Any] | None
    created_at: datetime


class SessionSummary(BaseModel):
    """A chat session as it appears in the Recents list (R8.1)."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str | None
    updated_at: datetime


class SessionDetail(BaseModel):
    """A chat session with its full ordered message transcript."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str | None
    selected_address_id: str | None
    messages: list[MessageResponse]
