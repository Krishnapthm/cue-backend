"""users (R2.0, CUE-6)."""

from __future__ import annotations

from sqlalchemy import BigInteger, CheckConstraint, Identity, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models.mixins import CreatedAtMixin, UpdatedAtMixin


class User(CreatedAtMixin, UpdatedAtMixin, Base):
    __tablename__ = "user"
    __table_args__ = (
        CheckConstraint("char_length(firebase_uid) <= 128", name="firebase_uid_length"),
        CheckConstraint("char_length(email) <= 320", name="email_length"),
        CheckConstraint("char_length(display_name) <= 200", name="display_name_length"),
    )

    # No public_id: the user id never appears in a URL. Every endpoint is scoped
    # to "me", resolved from the Firebase ID token via firebase_uid.
    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    firebase_uid: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    # email is deliberately NOT unique: firebase_uid is the identity. Firebase can
    # present the same email across providers, and a UNIQUE here would reject a
    # legitimate sign-in.
    email: Mapped[str] = mapped_column(Text, nullable=False)
    display_name: Mapped[str | None] = mapped_column(Text)
