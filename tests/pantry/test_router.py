from __future__ import annotations

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.pantry import PantryItem
from app.models.user import User
from app.pantry import service
from app.pantry.constants import CATEGORY_DISPLAY_ORDER, PantryCategory
from app.pantry.schemas import PantryItemCreate

RICE = {"name": "Basmati Rice", "category": "Grains & pulses"}


async def test_list_requires_authentication(client: httpx.AsyncClient) -> None:
    response = await client.get("/pantry")

    assert response.status_code == 401


async def test_write_routes_require_authentication(client: httpx.AsyncClient) -> None:
    assert (await client.post("/pantry", json=RICE)).status_code == 401
    assert (await client.patch("/pantry/1", json={"level": 0})).status_code == 401
    assert (await client.delete("/pantry/1")).status_code == 401


async def test_list_is_empty_on_day_one(authed_client: httpx.AsyncClient) -> None:
    response = await authed_client.get("/pantry")

    assert response.status_code == 200
    assert response.json() == []


async def test_add_creates_an_item_defaulted_to_full(
    authed_client: httpx.AsyncClient,
) -> None:
    response = await authed_client.post("/pantry", json=RICE)

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "Basmati Rice"
    assert body["category"] == "Grains & pulses"
    assert body["level"] == 3
    assert body["last_bought_at"] is None
    # `name_normalized` is an internal matching key, not part of the contract.
    assert "name_normalized" not in body


async def test_add_of_a_known_name_updates_rather_than_duplicating(
    authed_client: httpx.AsyncClient,
) -> None:
    first = await authed_client.post("/pantry", json=RICE)
    repeat = await authed_client.post(
        "/pantry",
        json={"name": "basmati rice", "category": "Grains & pulses", "level": 1},
    )

    assert repeat.status_code == 201
    assert repeat.json()["id"] == first.json()["id"]
    assert repeat.json()["level"] == 1

    listed = await authed_client.get("/pantry")
    assert len(listed.json()) == 1


async def test_list_returns_categories_in_contract_order(
    authed_client: httpx.AsyncClient,
) -> None:
    for category in reversed(CATEGORY_DISPLAY_ORDER):
        response = await authed_client.post(
            "/pantry",
            json={"name": f"item {category.value}", "category": category.value},
        )
        assert response.status_code == 201

    listed = await authed_client.get("/pantry")

    assert [item["category"] for item in listed.json()] == [
        category.value for category in CATEGORY_DISPLAY_ORDER
    ]


async def test_add_rejects_a_category_outside_the_fixed_set(
    authed_client: httpx.AsyncClient,
) -> None:
    response = await authed_client.post(
        "/pantry", json={"name": "Ghee", "category": "Condiments"}
    )

    assert response.status_code == 422


async def test_add_rejects_a_level_off_the_scale(
    authed_client: httpx.AsyncClient,
) -> None:
    for level in (-1, 4):
        response = await authed_client.post("/pantry", json={**RICE, "level": level})
        assert response.status_code == 422


async def test_add_rejects_a_blank_name(authed_client: httpx.AsyncClient) -> None:
    response = await authed_client.post(
        "/pantry", json={"name": "   ", "category": "Grains & pulses"}
    )

    assert response.status_code == 422


async def test_patch_writes_the_level(
    authed_client: httpx.AsyncClient, pantry_item: PantryItem
) -> None:
    response = await authed_client.patch(f"/pantry/{pantry_item.id}", json={"level": 0})

    assert response.status_code == 200
    assert response.json()["level"] == 0
    assert response.json()["name"] == "Basmati Rice"


async def test_patch_rejects_a_level_off_the_scale(
    authed_client: httpx.AsyncClient, pantry_item: PantryItem
) -> None:
    response = await authed_client.patch(f"/pantry/{pantry_item.id}", json={"level": 9})

    assert response.status_code == 422


async def test_patch_rejects_a_body_that_changes_nothing(
    authed_client: httpx.AsyncClient, pantry_item: PantryItem
) -> None:
    assert (
        await authed_client.patch(f"/pantry/{pantry_item.id}", json={})
    ).status_code == 422
    assert (
        await authed_client.patch(f"/pantry/{pantry_item.id}", json={"level": None})
    ).status_code == 422


async def test_patch_conflicts_when_a_rename_collides(
    authed_client: httpx.AsyncClient, db_session: AsyncSession, user: User
) -> None:
    rice = await service.upsert_item(
        db_session,
        user.id,
        PantryItemCreate(
            name="Basmati Rice", category=PantryCategory.GRAINS_AND_PULSES
        ),
    )
    dal = await service.upsert_item(
        db_session,
        user.id,
        PantryItemCreate(name="Toor Dal", category=PantryCategory.GRAINS_AND_PULSES),
    )
    # Read the ids before the request: the conflict rolls the session back,
    # which expires these instances, and re-reading an attribute afterwards
    # would trigger lazy IO outside the async context.
    rice_id, dal_id = rice.id, dal.id
    assert rice_id != dal_id

    response = await authed_client.patch(
        f"/pantry/{dal_id}", json={"name": "BASMATI RICE"}
    )

    assert response.status_code == 409
    # The rejected rename left both items exactly as they were.
    listed = (await authed_client.get("/pantry")).json()
    assert sorted(item["name"] for item in listed) == ["Basmati Rice", "Toor Dal"]


async def test_delete_removes_the_item(
    authed_client: httpx.AsyncClient, pantry_item: PantryItem
) -> None:
    response = await authed_client.delete(f"/pantry/{pantry_item.id}")

    assert response.status_code == 204
    assert (await authed_client.get("/pantry")).json() == []


async def test_delete_of_an_unknown_item_is_a_404(
    authed_client: httpx.AsyncClient,
) -> None:
    assert (await authed_client.delete("/pantry/999999")).status_code == 404


async def test_one_user_can_neither_read_nor_mutate_anothers_pantry(
    authed_client: httpx.AsyncClient, db_session: AsyncSession, other_user: User
) -> None:
    theirs = await service.upsert_item(
        db_session,
        other_user.id,
        PantryItemCreate(name="Their Rice", category=PantryCategory.GRAINS_AND_PULSES),
    )

    assert (await authed_client.get("/pantry")).json() == []
    # 404, not 403: the caller must not learn that this id exists at all.
    assert (
        await authed_client.patch(f"/pantry/{theirs.id}", json={"level": 0})
    ).status_code == 404
    assert (await authed_client.delete(f"/pantry/{theirs.id}")).status_code == 404

    await db_session.refresh(theirs)
    assert theirs.level == 3
