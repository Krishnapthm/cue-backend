"""Shared timestamp columns.

`created_at` lives on every table. `updated_at` lives only on the mutable ones
and is kept current by the shared `set_updated_at()` trigger (defined in the
initial migration), not by the ORM - a DB-level default is the single source of
truth regardless of who issues the UPDATE.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, text
from sqlalchemy.orm import Mapped, mapped_column


class CreatedAtMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("now()"),
    )


class UpdatedAtMixin:
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("now()"),
    )
