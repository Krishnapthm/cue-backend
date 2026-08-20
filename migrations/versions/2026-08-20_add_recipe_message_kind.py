"""add recipe message kind

Revision ID: d52b9c1f4a37
Revises: c41f7de9a8b2
Create Date: 2026-08-20 07:15:00.000000

Widens `ck_chat_message_kind_allowed` to admit a fifth kind, `recipe`, so the
generated recipe can be persisted into the display transcript once the cart is
ready (CUE-118). The recipe lives only inside the LangGraph checkpoint today -
an opaque blob keyed by thread - and cooking mode runs hours later, after a
cold start, off the transcript.

`chat_message.payload` is unchanged: it is still never filtered, joined, or
aggregated on, so admitting one more kind of card into it needs no index and no
column.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d52b9c1f4a37"
down_revision: str | Sequence[str] | None = "c41f7de9a8b2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# The *bare* constraint name. `MetaData`'s naming convention renders it as
# `ck_chat_message_kind_allowed`, and `op.drop_constraint` /
# `op.create_check_constraint` both apply that convention themselves - passing
# the rendered name here would double the prefix.
_CONSTRAINT = "kind_allowed"
_TABLE = "chat_message"

_KINDS_BEFORE = "'text', 'image', 'checklist', 'cart_ready'"
_KINDS_AFTER = "'text', 'image', 'checklist', 'cart_ready', 'recipe'"


def upgrade() -> None:
    """Admit `recipe` as a chat message kind."""
    op.drop_constraint(_CONSTRAINT, _TABLE, type_="check")
    op.create_check_constraint(_CONSTRAINT, _TABLE, f"kind IN ({_KINDS_AFTER})")


def downgrade() -> None:
    """Narrow the constraint back, dropping the rows it would now reject.

    The rows have to go before the constraint is re-added, or the `ADD
    CONSTRAINT` fails against any database a recipe turn has ever run on and
    the downgrade is not reversible at all.

    Deleting them is the right call rather than raising: `chat_message` is the
    *display* transcript, and the recipe these rows carry still exists in the
    LangGraph checkpoint the turn wrote it from. A downgrade that refused to
    run until an operator hand-deleted rows would be a downgrade nobody can
    execute during an incident, which is the one moment it exists for.
    """
    op.execute(f"DELETE FROM {_TABLE} WHERE kind = 'recipe'")
    op.drop_constraint(_CONSTRAINT, _TABLE, type_="check")
    op.create_check_constraint(_CONSTRAINT, _TABLE, f"kind IN ({_KINDS_BEFORE})")
