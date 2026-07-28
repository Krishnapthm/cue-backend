"""Pantry persistence and the last_bought_at stamp (CUE-69).

Every read and write in here is scoped by `user_id`. There is no unscoped
lookup by primary key on purpose - the only way to reach an item is through
a query that also constrains the owner, so a caller cannot read or mutate
another user's pantry even with a valid item id.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, cast

from sqlalchemy import CursorResult, case, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.cart import CartPlanItem
from app.models.pantry import PantryItem
from app.pantry.constants import CATEGORY_DISPLAY_ORDER, LEVEL_MIN
from app.pantry.exceptions import PantryItemNameConflictError, PantryItemNotFoundError
from app.pantry.schemas import PantryItemCreate, PantryItemUpdate

logger = logging.getLogger(__name__)

# Maps each category onto its position in the contract's display order, so the
# ordering lives in the query rather than being re-sorted in Python after the
# fact. Built from the enum, so adding a category cannot forget to update it.
_CATEGORY_POSITION = case(
    {
        category.value: position
        for position, category in enumerate(CATEGORY_DISPLAY_ORDER)
    },
    value=PantryItem.category,
)


def normalize_name(raw: str) -> str:
    """Return the form a pantry item's name is matched on.

    Case-folded and whitespace-collapsed, so "Basmati  Rice", "basmati rice"
    and " BASMATI RICE " are all one staple. This is the single definition of
    the rule: it is applied in Python and stored in `name_normalized` rather
    than computed by a database expression, so uniqueness, the `POST` upsert,
    and the `last_bought_at` match can never disagree about what "the same
    item" means.

    Args:
        raw: The name as the user typed it.

    Returns:
        The normalized form. Empty if `raw` held no non-whitespace characters.
    """
    return " ".join(raw.split()).casefold()


async def stocked_names(session: AsyncSession, user_id: int) -> set[str]:
    """Return the normalized names of the staples the user still has in stock.

    "In stock" is `level > 0`: the 0-3 scale's zero means Out, and an item the
    user has marked Out is exactly the one they still need bought. The read
    selects the single column it needs rather than hydrating `PantryItem` rows,
    since the caller only ever asks the set for membership.

    Returned in the normalized form so the caller compares like with like -
    `normalize_name` is the one definition of "the same item" in this codebase,
    and matching on `name` would make "Basmati Rice" and "basmati rice"
    different staples again.

    Args:
        session: An active database session.
        user_id: The owning Cue user.

    Returns:
        The normalized names of every in-stock staple. Empty for a user who
        keeps no pantry, which is the normal state for a new account.
    """
    stmt = select(PantryItem.name_normalized).where(
        PantryItem.user_id == user_id, PantryItem.level > LEVEL_MIN
    )
    result = await session.execute(stmt)
    return set(result.scalars().all())


async def list_items(session: AsyncSession, user_id: int) -> list[PantryItem]:
    """Return the user's pantry in the contract's category order.

    Ordering within a category is the client's concern, but a stable
    tiebreaker is applied anyway so the response does not reshuffle between
    identical requests.

    Args:
        session: An active database session.
        user_id: The owning Cue user.

    Returns:
        The user's items, grouped in category display order. An empty list is
        the normal state for a new account, not an error.
    """
    stmt = (
        select(PantryItem)
        .where(PantryItem.user_id == user_id)
        .order_by(_CATEGORY_POSITION, PantryItem.name_normalized)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def get_item(session: AsyncSession, user_id: int, item_id: int) -> PantryItem:
    """Return one pantry item, scoped to its owner.

    Args:
        session: An active database session.
        user_id: The Cue user who must own `item_id`.
        item_id: The item to fetch.

    Returns:
        The matching item.

    Raises:
        PantryItemNotFoundError: If `item_id` does not exist, or exists but
            belongs to another user - both are the same 404.
    """
    stmt = select(PantryItem).where(
        PantryItem.id == item_id, PantryItem.user_id == user_id
    )
    result = await session.execute(stmt)
    item = result.scalar_one_or_none()
    if item is None:
        raise PantryItemNotFoundError
    return item


async def upsert_item(
    session: AsyncSession, user_id: int, request: PantryItemCreate
) -> PantryItem:
    """Add a pantry item, or update the existing one with the same name.

    Idempotent against the uniqueness constraint by construction: a single
    `INSERT ... ON CONFLICT DO UPDATE` rather than a select-then-insert,
    which would race two concurrent adds of the same staple into an
    IntegrityError. A repeat add rewrites `name` too, so the stored display
    form follows the casing the user typed most recently.

    Args:
        session: An active database session.
        user_id: The owning Cue user.
        request: The item to add; `level` already defaulted to full by the
            schema when the client omitted it.

    Returns:
        The created or updated item.
    """
    stmt = (
        pg_insert(PantryItem)
        .values(
            user_id=user_id,
            name=request.name,
            name_normalized=normalize_name(request.name),
            category=request.category.value,
            level=request.level,
        )
        .on_conflict_do_update(
            constraint="uq_pantry_item_user_name",
            set_={
                "name": request.name,
                "category": request.category.value,
                "level": request.level,
            },
        )
        .returning(PantryItem)
        # On the conflict branch the row is already in the session's identity
        # map from an earlier read, and SQLAlchemy will not overwrite a live
        # object's attributes by default - without this, a repeat add returns
        # the stale pre-update values even though the UPDATE did happen.
        .execution_options(populate_existing=True)
    )
    result = await session.execute(stmt)
    item = result.scalar_one()
    await session.commit()
    return item


async def update_item(
    session: AsyncSession, item: PantryItem, request: PantryItemUpdate
) -> PantryItem:
    """Apply a partial update to an item the caller already owns.

    Only fields the request actually carries are written; the schema rejects
    a body that would change nothing, so `None` here means "left alone".
    Ownership is not rechecked - `item` comes from `owned_pantry_item`,
    which already scoped the lookup by user.

    Args:
        session: An active database session.
        item: The caller's item, as resolved by the route dependency.
        request: The fields to change.

    Returns:
        The updated item.

    Raises:
        PantryItemNameConflictError: If a rename collides with another item
            already in the caller's pantry.
    """
    if request.name is not None:
        item.name = request.name
        item.name_normalized = normalize_name(request.name)
    if request.category is not None:
        item.category = request.category.value
    if request.level is not None:
        item.level = request.level

    try:
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        raise PantryItemNameConflictError from exc

    await session.refresh(item)
    return item


async def delete_item(session: AsyncSession, item: PantryItem) -> None:
    """Remove an item the caller already owns.

    Args:
        session: An active database session.
        item: The caller's item, as resolved by the route dependency.
    """
    await session.delete(item)
    await session.commit()


async def stamp_last_bought(
    session: AsyncSession, user_id: int, plan_id: int, bought_at: datetime
) -> int:
    """Stamp `last_bought_at` on every pantry item the placed order contained.

    Matching is on the normalized name, against the plan's ingredient names
    rather than the matched Swiggy product names: the pantry holds staples
    ("basmati rice"), while the product name is a specific pack ("India Gate
    Basmati Rice 5kg"), which would almost never match.

    The name list is bounded by the size of one recipe's ingredient list, so
    it is a small `IN (...)`, not an unbounded one built from client input.

    This does not commit - the caller owns the transaction boundary, so a
    stamp can be rolled back independently of the order it belongs to.

    Args:
        session: An active database session.
        user_id: The user whose pantry to stamp.
        plan_id: The cart plan the placed order was built from.
        bought_at: The order's placement time, so the stamp matches the order
            rather than whenever this bookkeeping happened to run.

    Returns:
        How many pantry items were stamped; zero when the user keeps no
        pantry, or none of the ordered ingredients are in it.
    """
    names_stmt = select(CartPlanItem.ingredient_name).where(
        CartPlanItem.plan_id == plan_id
    )
    result = await session.execute(names_stmt)
    normalized = {normalize_name(name) for name in result.scalars().all()}
    normalized.discard("")
    if not normalized:
        return 0

    stamp_stmt = (
        update(PantryItem)
        .where(
            PantryItem.user_id == user_id,
            PantryItem.name_normalized.in_(normalized),
        )
        .values(last_bought_at=bought_at)
    )
    stamped = cast("CursorResult[Any]", await session.execute(stamp_stmt))
    return stamped.rowcount
