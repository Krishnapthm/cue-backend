from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.cart import CartPlan, CartPlanItem
from app.models.user import User
from app.pantry import service
from app.pantry.constants import CATEGORY_DISPLAY_ORDER, PantryCategory
from app.pantry.exceptions import PantryItemNameConflictError, PantryItemNotFoundError
from app.pantry.models import PantryItem
from app.pantry.schemas import PantryItemCreate, PantryItemUpdate


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Basmati Rice", "basmati rice"),
        ("  basmati   rice  ", "basmati rice"),
        ("BASMATI\tRICE", "basmati rice"),
        ("Toor Dal", "toor dal"),
        ("   ", ""),
    ],
)
def test_normalize_name_folds_case_and_collapses_whitespace(
    raw: str, expected: str
) -> None:
    assert service.normalize_name(raw) == expected


async def test_list_items_returns_categories_in_contract_order(
    db_session: AsyncSession, user: User
) -> None:
    # Insert in reverse contract order, so passing cannot be an accident of
    # insertion order or of the primary key.
    for category in reversed(CATEGORY_DISPLAY_ORDER):
        await service.upsert_item(
            db_session,
            user.id,
            PantryItemCreate(name=f"item {category.value}", category=category),
        )

    items = await service.list_items(db_session, user.id)

    assert [item.category for item in items] == [
        category.value for category in CATEGORY_DISPLAY_ORDER
    ]


async def test_list_items_is_empty_for_a_new_user(
    db_session: AsyncSession, user: User
) -> None:
    assert await service.list_items(db_session, user.id) == []


async def test_list_items_never_returns_another_users_pantry(
    db_session: AsyncSession, user: User, other_user: User
) -> None:
    await service.upsert_item(
        db_session,
        other_user.id,
        PantryItemCreate(name="Their Rice", category=PantryCategory.GRAINS_AND_PULSES),
    )

    assert await service.list_items(db_session, user.id) == []


async def test_upsert_item_defaults_level_to_full(
    db_session: AsyncSession, user: User
) -> None:
    item = await service.upsert_item(
        db_session,
        user.id,
        PantryItemCreate(
            name="Basmati Rice", category=PantryCategory.GRAINS_AND_PULSES
        ),
    )

    assert item.level == 3
    assert item.name_normalized == "basmati rice"
    assert item.last_bought_at is None


async def test_upsert_item_updates_instead_of_duplicating_a_known_name(
    db_session: AsyncSession, user: User
) -> None:
    first = await service.upsert_item(
        db_session,
        user.id,
        PantryItemCreate(
            name="Basmati Rice", category=PantryCategory.GRAINS_AND_PULSES
        ),
    )

    # Same staple, typed differently, in a different category, nearly empty.
    second = await service.upsert_item(
        db_session,
        user.id,
        PantryItemCreate(
            name="  basmati   RICE ",
            category=PantryCategory.SNACKS_AND_PACKAGED,
            level=1,
        ),
    )

    assert second.id == first.id
    assert second.level == 1
    assert second.category == PantryCategory.SNACKS_AND_PACKAGED.value
    # The display form follows the casing typed most recently.
    assert second.name == "basmati   RICE"
    assert len(await service.list_items(db_session, user.id)) == 1


async def test_upsert_item_keeps_two_users_pantries_separate(
    db_session: AsyncSession, user: User, other_user: User
) -> None:
    mine = await service.upsert_item(
        db_session,
        user.id,
        PantryItemCreate(
            name="Basmati Rice", category=PantryCategory.GRAINS_AND_PULSES
        ),
    )
    theirs = await service.upsert_item(
        db_session,
        other_user.id,
        PantryItemCreate(
            name="Basmati Rice", category=PantryCategory.GRAINS_AND_PULSES
        ),
    )

    # Same name is not a conflict across users - uniqueness is per pantry.
    assert mine.id != theirs.id


async def test_get_item_rejects_another_users_item(
    db_session: AsyncSession, other_user: User, pantry_item: PantryItem
) -> None:
    with pytest.raises(PantryItemNotFoundError):
        await service.get_item(db_session, other_user.id, pantry_item.id)


async def test_update_item_writes_only_the_fields_supplied(
    db_session: AsyncSession, pantry_item: PantryItem
) -> None:
    updated = await service.update_item(
        db_session, pantry_item, PantryItemUpdate(level=0)
    )

    assert updated.level == 0
    assert updated.name == "Basmati Rice"
    assert updated.category == PantryCategory.GRAINS_AND_PULSES.value


async def test_update_item_renormalizes_on_rename(
    db_session: AsyncSession, pantry_item: PantryItem
) -> None:
    updated = await service.update_item(
        db_session, pantry_item, PantryItemUpdate(name="  Sona  Masoori ")
    )

    assert updated.name == "Sona  Masoori"
    assert updated.name_normalized == "sona masoori"


async def test_update_item_rejects_a_rename_onto_another_item(
    db_session: AsyncSession, user: User, pantry_item: PantryItem
) -> None:
    other = await service.upsert_item(
        db_session,
        user.id,
        PantryItemCreate(name="Toor Dal", category=PantryCategory.GRAINS_AND_PULSES),
    )

    with pytest.raises(PantryItemNameConflictError):
        await service.update_item(
            db_session, other, PantryItemUpdate(name="basmati rice")
        )


async def test_delete_item_removes_it(
    db_session: AsyncSession, user: User, pantry_item: PantryItem
) -> None:
    await service.delete_item(db_session, pantry_item)

    assert await service.list_items(db_session, user.id) == []


async def _plan_with_ingredients(
    db_session: AsyncSession, plan: CartPlan, *names: str
) -> None:
    for name in names:
        db_session.add(
            CartPlanItem(
                plan_id=plan.id,
                ingredient_name=name,
                match_status="unavailable",
            )
        )
    await db_session.commit()


async def test_stamp_last_bought_marks_matching_items_only(
    db_session: AsyncSession, user: User, cart_plan: CartPlan
) -> None:
    rice = await service.upsert_item(
        db_session,
        user.id,
        PantryItemCreate(
            name="Basmati Rice", category=PantryCategory.GRAINS_AND_PULSES
        ),
    )
    salt = await service.upsert_item(
        db_session,
        user.id,
        PantryItemCreate(name="Salt", category=PantryCategory.SPICES_AND_MASALAS),
    )
    # The plan names the rice with different casing and spacing, and names an
    # ingredient the pantry does not track at all.
    await _plan_with_ingredients(db_session, cart_plan, "  BASMATI  rice ", "Paneer")

    bought_at = datetime(2026, 7, 26, 12, 0, tzinfo=UTC)
    stamped = await service.stamp_last_bought(
        db_session, user.id, cart_plan.id, bought_at
    )
    await db_session.commit()

    assert stamped == 1
    await db_session.refresh(rice)
    await db_session.refresh(salt)
    assert rice.last_bought_at == bought_at
    assert salt.last_bought_at is None


async def test_stamp_last_bought_never_touches_another_users_pantry(
    db_session: AsyncSession, user: User, other_user: User, cart_plan: CartPlan
) -> None:
    theirs = await service.upsert_item(
        db_session,
        other_user.id,
        PantryItemCreate(
            name="Basmati Rice", category=PantryCategory.GRAINS_AND_PULSES
        ),
    )
    await _plan_with_ingredients(db_session, cart_plan, "Basmati Rice")

    stamped = await service.stamp_last_bought(
        db_session, user.id, cart_plan.id, datetime.now(UTC)
    )
    await db_session.commit()

    assert stamped == 0
    await db_session.refresh(theirs)
    assert theirs.last_bought_at is None


async def test_stamp_last_bought_is_a_no_op_for_an_empty_plan(
    db_session: AsyncSession, user: User, cart_plan: CartPlan
) -> None:
    assert (
        await service.stamp_last_bought(
            db_session, user.id, cart_plan.id, datetime.now(UTC)
        )
        == 0
    )


async def test_level_outside_the_scale_is_refused_by_the_database(
    db_session: AsyncSession, user: User
) -> None:
    """The 0-3 scale holds even if a caller bypasses the Pydantic schema."""
    db_session.add(
        PantryItem(
            user_id=user.id,
            name="Ghee",
            name_normalized="ghee",
            category=PantryCategory.DAIRY_AND_EGGS.value,
            level=4,
        )
    )
    with pytest.raises(IntegrityError):
        await db_session.commit()
    await db_session.rollback()


async def test_unknown_category_is_refused_by_the_database(
    db_session: AsyncSession, user: User
) -> None:
    db_session.add(
        PantryItem(
            user_id=user.id,
            name="Ghee",
            name_normalized="ghee",
            category="Condiments",
            level=3,
        )
    )
    with pytest.raises(IntegrityError):
        await db_session.commit()
    await db_session.rollback()


async def test_deleting_a_user_cascades_to_their_pantry(
    db_session: AsyncSession, user: User, pantry_item: PantryItem
) -> None:
    user_id = user.id
    await db_session.delete(user)
    await db_session.commit()

    # Scoped to this user: the test database is shared across the session and
    # is never truncated between tests, so other users' rows are still there.
    remaining = await db_session.execute(
        select(PantryItem).where(PantryItem.user_id == user_id)
    )
    assert remaining.scalars().all() == []
