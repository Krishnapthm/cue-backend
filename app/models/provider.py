"""provider_links and oauth_transactions (R2.1/R2.5/R9.2/R9.3, CUE-7/8/9)."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Identity,
    LargeBinary,
    SmallInteger,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models.mixins import CreatedAtMixin, UpdatedAtMixin


class ProviderLink(CreatedAtMixin, UpdatedAtMixin, Base):
    __tablename__ = "provider_link"
    __table_args__ = (
        # UNIQUE(user_id, provider) doubles as the FK index on user_id.
        UniqueConstraint("user_id", "provider", name="uq_provider_link_user_provider"),
        # provider column (not a swiggy_link table) so the adapter seam is a
        # one-line CHECK swap when Zepto/Blinkit land.
        CheckConstraint("provider IN ('swiggy')", name="provider_allowed"),
        CheckConstraint("char_length(scope) <= 500", name="scope_length"),
        CheckConstraint("status IN ('active', 'expired')", name="status_allowed"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("user.id", ondelete="CASCADE"), nullable=False
    )
    provider: Mapped[str] = mapped_column(Text, nullable=False)
    # AES-GCM/Fernet ciphertext encrypted in the app (not pgcrypto). key_version
    # supports rotation. No refresh_token column: Swiggy v1.0 issues access
    # tokens only. Revisit at Swiggy v1.1.
    access_token_ct: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    key_version: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, server_default=text("1")
    )
    token_expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    scope: Mapped[str] = mapped_column(Text, nullable=False)
    # 'expired' is distinct from expiry-by-clock: Swiggy can revoke server-side
    # before exp (a 401/419 mid-call), which the clock cannot see.
    status: Mapped[str] = mapped_column(Text, nullable=False)
    linked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class OAuthTransaction(CreatedAtMixin, Base):
    """In-flight PKCE, transient. Hard-deleted on a sweep once consumed/expired."""

    __tablename__ = "oauth_transaction"
    __table_args__ = (
        CheckConstraint("char_length(state) <= 128", name="state_length"),
        CheckConstraint("provider IN ('swiggy')", name="provider_allowed"),
        CheckConstraint(
            "char_length(redirect_uri) <= 2000", name="redirect_uri_length"
        ),
    )

    state: Mapped[str] = mapped_column(Text, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("user.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    provider: Mapped[str] = mapped_column(Text, nullable=False)
    # The PKCE verifier is a live secret for the 120s the auth code is valid.
    code_verifier_ct: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    key_version: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, server_default=text("1")
    )
    redirect_uri: Mapped[str] = mapped_column(Text, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    consumed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
