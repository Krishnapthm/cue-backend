"""chat_sessions and chat_messages (R8.3, CUE-20/45).

chat_sessions.id IS the LangGraph thread_id (str(session.id)). There is no FK
from checkpoints.thread_id to here - different ownership, different lifecycle -
so deleting a session must also call checkpointer.adelete_thread(id).
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    ForeignKey,
    Identity,
    Index,
    Text,
    Uuid,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models.mixins import CreatedAtMixin, UpdatedAtMixin


class ChatSession(CreatedAtMixin, UpdatedAtMixin, Base):
    __tablename__ = "chat_session"
    __table_args__ = (
        CheckConstraint("char_length(title) <= 200", name="title_length"),
        CheckConstraint(
            "char_length(selected_address_id) <= 100",
            name="selected_address_id_length",
        ),
        # Recents (R8.1) is a flat title-only list ordered by updated_at DESC.
        Index("ix_chat_session_user_recents", "user_id", text("updated_at DESC")),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, server_default=text("gen_random_uuid()")
    )
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("user.id", ondelete="CASCADE"), nullable=False
    )
    # Nullable: a session exists before the agent has titled it. Client renders
    # a fallback.
    title: Mapped[str | None] = mapped_column(Text)
    # The Swiggy addressId, not an FK (we do not own addresses). The chat
    # header's "Delivering / Home" state.
    selected_address_id: Mapped[str | None] = mapped_column(Text)


class ChatMessage(CreatedAtMixin, Base):
    """Display transcript. Duplicates the checkpointer on purpose - the
    checkpointer holds opaque serialized blobs; this is fast and queryable."""

    __tablename__ = "chat_message"
    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant')", name="role_allowed"),
        CheckConstraint(
            "kind IN ('text', 'image', 'checklist', 'cart_ready')",
            name="kind_allowed",
        ),
        CheckConstraint(
            "(kind = 'text' AND content IS NOT NULL)"
            " OR (kind <> 'text' AND payload IS NOT NULL)",
            name="body",
        ),
        # Ordering is by id, not created_at: identity is monotonic and
        # collision-free. (session_id, id) also serves keyset pagination.
        Index("ix_chat_message_session", "session_id", "id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    session_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("chat_session.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(Text, nullable=False)
    kind: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'text'")
    )
    content: Mapped[str | None] = mapped_column(Text)
    # kind='image': the Supabase Storage object path (image bytes are not stored
    # in Postgres). kind='checklist'/'cart_ready': the rendered card. Never
    # filtered, joined, or aggregated on - anything we query lives in
    # cart_plan_items.
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
