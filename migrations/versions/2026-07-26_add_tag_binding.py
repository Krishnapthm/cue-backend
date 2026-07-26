"""add tag binding

Revision ID: c41f7de9a8b2
Revises: b70a3b3b2ee3
Create Date: 2026-07-26 18:40:00.000000

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c41f7de9a8b2"
down_revision: str | Sequence[str] | None = "b70a3b3b2ee3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "tag_binding",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("tag_uid", sa.Text(), nullable=False),
        sa.Column("tag_text", sa.Text(), nullable=False),
        sa.Column("spin_id", sa.Text(), nullable=False),
        sa.Column("product_id", sa.Text(), nullable=True),
        sa.Column("product_name", sa.Text(), nullable=True),
        sa.Column("refill_size", sa.Text(), nullable=True),
        sa.Column("unit_price", sa.Numeric(precision=10, scale=2), nullable=True),
        sa.Column("address_id", sa.Text(), nullable=False),
        sa.Column("pantry_item_id", sa.BigInteger(), nullable=True),
        sa.Column(
            "last_used_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "char_length(address_id) <= 100",
            name=op.f("ck_tag_binding_address_id_length"),
        ),
        sa.CheckConstraint(
            "char_length(product_id) <= 100",
            name=op.f("ck_tag_binding_product_id_length"),
        ),
        sa.CheckConstraint(
            "char_length(product_name) <= 300",
            name=op.f("ck_tag_binding_product_name_length"),
        ),
        sa.CheckConstraint(
            "char_length(refill_size) <= 50",
            name=op.f("ck_tag_binding_refill_size_length"),
        ),
        sa.CheckConstraint(
            "char_length(spin_id) <= 100", name=op.f("ck_tag_binding_spin_id_length")
        ),
        sa.CheckConstraint(
            "char_length(tag_text) <= 200", name=op.f("ck_tag_binding_tag_text_length")
        ),
        sa.CheckConstraint(
            "char_length(tag_uid) <= 100", name=op.f("ck_tag_binding_tag_uid_length")
        ),
        sa.CheckConstraint(
            "tag_text <> ''", name=op.f("ck_tag_binding_tag_text_not_blank")
        ),
        sa.CheckConstraint(
            "tag_uid <> ''", name=op.f("ck_tag_binding_tag_uid_not_blank")
        ),
        sa.CheckConstraint(
            "unit_price >= 0", name=op.f("ck_tag_binding_unit_price_nonneg")
        ),
        sa.ForeignKeyConstraint(
            ["pantry_item_id"],
            ["pantry_item.id"],
            name=op.f("fk_tag_binding_pantry_item_id_pantry_item"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["user.id"],
            name=op.f("fk_tag_binding_user_id_user"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_tag_binding")),
        sa.UniqueConstraint("user_id", "tag_uid", name="uq_tag_binding_user_tag"),
    )
    # No updated_at column, so tag_binding deliberately does not join the
    # shared set_updated_at() trigger: `last_used_at` is written explicitly by
    # the resolver and means "last scanned", not "last modified".


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("tag_binding")
