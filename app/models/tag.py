"""tag_binding (CUE-74) - what an NFC sticker currently resolves to.

An NFC tag is written once, physically, and never re-written, so it can only
ever carry a stable human word ("sugar") - never a `spin_id`, which has a
shelf life. The binding from that word to a concrete, orderable Instamart
variant therefore lives here, server-side, and is re-derived whenever it stops
being usable.

`spin_id` is not nullable: Swiggy's `update_cart` addresses items by `spinId`,
so a row without one could not build a cart line and is not worth caching.
Nothing purchasable is simply not persisted at all.

`address_id` is part of what makes a binding valid, not decoration. Instamart
search results and stock are address-scoped, so a variant bound while ordering
to one address may be unorderable at another; a request for a different
address is treated as a miss and re-resolved rather than returning a variant
that would die at checkout.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Identity,
    Numeric,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models.mixins import CreatedAtMixin
from app.tags.constants import (
    MAX_ADDRESS_ID_LENGTH,
    MAX_PRODUCT_ID_LENGTH,
    MAX_PRODUCT_NAME_LENGTH,
    MAX_REFILL_SIZE_LENGTH,
    MAX_SPIN_ID_LENGTH,
    MAX_TAG_TEXT_LENGTH,
    MAX_TAG_UID_LENGTH,
)


class TagBinding(CreatedAtMixin, Base):
    __tablename__ = "tag_binding"
    __table_args__ = (
        CheckConstraint(
            f"char_length(tag_uid) <= {MAX_TAG_UID_LENGTH}", name="tag_uid_length"
        ),
        CheckConstraint("tag_uid <> ''", name="tag_uid_not_blank"),
        CheckConstraint(
            f"char_length(tag_text) <= {MAX_TAG_TEXT_LENGTH}", name="tag_text_length"
        ),
        CheckConstraint("tag_text <> ''", name="tag_text_not_blank"),
        CheckConstraint(
            f"char_length(spin_id) <= {MAX_SPIN_ID_LENGTH}", name="spin_id_length"
        ),
        CheckConstraint(
            f"char_length(product_id) <= {MAX_PRODUCT_ID_LENGTH}",
            name="product_id_length",
        ),
        CheckConstraint(
            f"char_length(product_name) <= {MAX_PRODUCT_NAME_LENGTH}",
            name="product_name_length",
        ),
        CheckConstraint(
            f"char_length(refill_size) <= {MAX_REFILL_SIZE_LENGTH}",
            name="refill_size_length",
        ),
        CheckConstraint(
            f"char_length(address_id) <= {MAX_ADDRESS_ID_LENGTH}",
            name="address_id_length",
        ),
        CheckConstraint("unit_price >= 0", name="unit_price_nonneg"),
        # A sticker belongs to exactly one household item, and bindings never
        # cross users. Leading with `user_id` also serves the foreign key and
        # every per-user read, so no separate index on `user_id` is needed.
        UniqueConstraint("user_id", "tag_uid", name="uq_tag_binding_user_tag"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("user.id", ondelete="CASCADE"), nullable=False
    )
    # The tag's hardware UID, as read by the phone.
    tag_uid: Mapped[str] = mapped_column(Text, nullable=False)
    # The bare slug physically written on the sticker ("sugar", "haldi").
    tag_text: Mapped[str] = mapped_column(Text, nullable=False)
    spin_id: Mapped[str] = mapped_column(Text, nullable=False)
    product_id: Mapped[str | None] = mapped_column(Text)
    product_name: Mapped[str | None] = mapped_column(Text)
    refill_size: Mapped[str | None] = mapped_column(Text)
    # Price at bind time, so a cache hit can still render a cart line without
    # a search. A snapshot for display only - Swiggy prices the actual order.
    unit_price: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    # The address this variant was found orderable at; see the module docstring.
    address_id: Mapped[str] = mapped_column(Text, nullable=False)
    # The pantry staple this sticker is on, when the slug names one. Nullable:
    # a tag can be bound before the matching pantry item exists, and deleting
    # the pantry item must not delete the binding.
    pantry_item_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("pantry_item.id", ondelete="SET NULL")
    )
    last_used_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )
