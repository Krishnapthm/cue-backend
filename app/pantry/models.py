"""pantry_item (CUE-69) - the user's own model of what they keep in stock.

Server-owned on purpose: `level` is user-authored state that has to survive a
reinstall and follow the account across devices, and it cannot be recomputed
from order history because the Swiggy order feed only reaches back 15 days.

`name` keeps what the user typed; `name_normalized` is what uniqueness is
enforced on, so "Basmati Rice" and "basmati rice" are the same staple. The
normalized form is written by `app.pantry.service.normalize_name` rather than
derived in the database, so exactly one implementation of the rule exists.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Identity,
    SmallInteger,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models.mixins import CreatedAtMixin, UpdatedAtMixin
from app.pantry.constants import (
    CATEGORY_DISPLAY_ORDER,
    DEFAULT_LEVEL,
    LEVEL_MAX,
    LEVEL_MIN,
    MAX_NAME_LENGTH,
)

# Rendered from the enum so the constraint and the API contract can never
# drift apart. No category contains a quote, so this needs no escaping.
_CATEGORY_LITERALS = ", ".join(
    f"'{category.value}'" for category in CATEGORY_DISPLAY_ORDER
)


class PantryItem(CreatedAtMixin, UpdatedAtMixin, Base):
    __tablename__ = "pantry_item"
    __table_args__ = (
        CheckConstraint(f"char_length(name) <= {MAX_NAME_LENGTH}", name="name_length"),
        CheckConstraint(
            f"char_length(name_normalized) <= {MAX_NAME_LENGTH}",
            name="name_normalized_length",
        ),
        CheckConstraint("name_normalized <> ''", name="name_normalized_not_blank"),
        CheckConstraint(f"category IN ({_CATEGORY_LITERALS})", name="category_allowed"),
        # The 0-3 ordinal scale is load-bearing for the segment-bar UI, so the
        # database refuses anything off it even if a caller bypasses the schema.
        CheckConstraint(
            f"level BETWEEN {LEVEL_MIN} AND {LEVEL_MAX}", name="level_in_range"
        ),
        # One staple per user, matched case- and whitespace-insensitively, so a
        # repeat add updates instead of duplicating. This index leads with
        # `user_id`, so it also serves the foreign key and every per-user read;
        # a separate index on `user_id` alone would be redundant.
        UniqueConstraint("user_id", "name_normalized", name="uq_pantry_item_user_name"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("user.id", ondelete="CASCADE"), nullable=False
    )
    # As the user typed it - this is what the Pantry screen displays.
    name: Mapped[str] = mapped_column(Text, nullable=False)
    # Case-folded, whitespace-collapsed form; uniqueness and the last_bought_at
    # match both run on this, never on `name`.
    name_normalized: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(Text, nullable=False)
    # 0 = Out, 1 = Low, 2 = Half, 3 = Full.
    level: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, server_default=text(str(DEFAULT_LEVEL))
    )
    # Stamped server-side when a placed order contains this staple; never
    # client-supplied. Null means "never bought through Cue", which is the
    # normal state for a hand-added item.
    last_bought_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
