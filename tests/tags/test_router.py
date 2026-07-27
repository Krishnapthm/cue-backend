from __future__ import annotations

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.tags import service
from app.tags.constants import MAX_TAPS_PER_BATCH
from app.tags.schemas import TagResolveBatchRequest, TagTap
from tests.conftest import InstamartToolCallStub
from tests.tags.conftest import ADDRESS_ID, product, search_result

MADHUR_SUGAR = product(
    product_id="p-sugar-madhur",
    name="Madhur Pure Sugar",
    spin_id="spin-madhur",
    price="62.00",
)

SCAN = {
    "addressId": ADDRESS_ID,
    "taps": [{"tagUid": "04a2", "text": "sugar", "quantity": 2}],
}


async def test_every_route_requires_authentication(client: httpx.AsyncClient) -> None:
    assert (
        await client.post("/pantry/tags/resolve-batch", json=SCAN)
    ).status_code == 401
    assert (
        await client.patch("/pantry/tags/04a2", json={"spinId": "s", "addressId": "a"})
    ).status_code == 401
    assert (await client.delete("/pantry/tags/04a2")).status_code == 401


async def test_resolve_batch_returns_one_entry_per_tap(
    authed_client: httpx.AsyncClient, instamart: InstamartToolCallStub
) -> None:
    instamart.configure_search_query("sugar", search_result(MADHUR_SUGAR))

    response = await authed_client.post("/pantry/tags/resolve-batch", json=SCAN)

    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0] == {
        "tag_uid": "04a2",
        "text": "sugar",
        "outcome": "bound",
        "spin_id": "spin-madhur",
        "product_id": "p-sugar-madhur",
        "product_name": "Madhur Pure Sugar",
        "refill_size": "1 kg",
        "unit_price": "62.00",
        "in_stock": True,
        "pantry_item_id": None,
        "quantity": 2,
    }


async def test_an_unresolvable_slug_is_a_200_not_an_error(
    authed_client: httpx.AsyncClient, instamart: InstamartToolCallStub
) -> None:
    instamart.configure_search_query("gulkand", search_result())

    response = await authed_client.post(
        "/pantry/tags/resolve-batch",
        json={"addressId": ADDRESS_ID, "taps": [{"tagUid": "04b7", "text": "gulkand"}]},
    )

    assert response.status_code == 200
    assert response.json()["results"][0]["outcome"] == "unresolved"
    assert response.json()["results"][0]["spin_id"] is None
    # Quantity defaults to one refill per tap.
    assert response.json()["results"][0]["quantity"] == 1


async def test_an_empty_or_oversized_batch_is_rejected(
    authed_client: httpx.AsyncClient,
) -> None:
    empty = await authed_client.post(
        "/pantry/tags/resolve-batch", json={"addressId": ADDRESS_ID, "taps": []}
    )
    oversized = await authed_client.post(
        "/pantry/tags/resolve-batch",
        json={
            "addressId": ADDRESS_ID,
            "taps": [
                {"tagUid": f"uid-{index}", "text": "sugar"}
                for index in range(MAX_TAPS_PER_BATCH + 1)
            ],
        },
    )

    assert empty.status_code == 422
    assert oversized.status_code == 422


async def test_a_tap_without_text_is_rejected(authed_client: httpx.AsyncClient) -> None:
    response = await authed_client.post(
        "/pantry/tags/resolve-batch",
        json={"addressId": ADDRESS_ID, "taps": [{"tagUid": "04a2", "text": "  "}]},
    )

    assert response.status_code == 422


async def test_patch_rebinds_a_tag_to_a_chosen_variant(
    authed_client: httpx.AsyncClient, instamart: InstamartToolCallStub
) -> None:
    instamart.configure_search_query("sugar", search_result(MADHUR_SUGAR))
    await authed_client.post("/pantry/tags/resolve-batch", json=SCAN)

    response = await authed_client.patch(
        "/pantry/tags/04a2",
        json={
            "spinId": "spin-chosen",
            "productName": "The One I Actually Buy",
            "refillSize": "2 kg",
            "unitPrice": "118.50",
            "addressId": ADDRESS_ID,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["spin_id"] == "spin-chosen"
    assert body["product_name"] == "The One I Actually Buy"
    assert body["refill_size"] == "2 kg"
    assert body["unit_price"] == "118.50"


async def test_a_rebound_tag_serves_the_chosen_variant_from_cache(
    authed_client: httpx.AsyncClient, instamart: InstamartToolCallStub
) -> None:
    instamart.configure_search_query("sugar", search_result(MADHUR_SUGAR))
    await authed_client.post("/pantry/tags/resolve-batch", json=SCAN)
    await authed_client.patch(
        "/pantry/tags/04a2",
        json={"spinId": "spin-chosen", "addressId": ADDRESS_ID},
    )

    response = await authed_client.post("/pantry/tags/resolve-batch", json=SCAN)

    assert response.json()["results"][0]["outcome"] == "cached"
    assert response.json()["results"][0]["spin_id"] == "spin-chosen"


async def test_delete_unbinds_so_the_next_scan_resolves_afresh(
    authed_client: httpx.AsyncClient, instamart: InstamartToolCallStub
) -> None:
    instamart.configure_search_query("sugar", search_result(MADHUR_SUGAR))
    await authed_client.post("/pantry/tags/resolve-batch", json=SCAN)

    deleted = await authed_client.delete("/pantry/tags/04a2")
    rescanned = await authed_client.post("/pantry/tags/resolve-batch", json=SCAN)

    assert deleted.status_code == 204
    assert rescanned.json()["results"][0]["outcome"] == "bound"


async def test_correcting_an_unknown_tag_is_a_404(
    authed_client: httpx.AsyncClient,
) -> None:
    patched = await authed_client.patch(
        "/pantry/tags/never-seen",
        json={"spinId": "spin-x", "addressId": ADDRESS_ID},
    )
    deleted = await authed_client.delete("/pantry/tags/never-seen")

    assert patched.status_code == 404
    assert deleted.status_code == 404


async def test_another_users_tag_uid_is_a_404_not_someone_elses_binding(
    authed_client: httpx.AsyncClient,
    db_session: AsyncSession,
    linked_other_user: User,
    instamart: InstamartToolCallStub,
) -> None:
    instamart.configure_search_query("sugar", search_result(MADHUR_SUGAR))
    await service.resolve_batch(
        db_session,
        linked_other_user.id,
        TagResolveBatchRequest(
            address_id=ADDRESS_ID,
            taps=[TagTap(tag_uid="04a2", text="sugar")],
        ),
    )

    response = await authed_client.delete("/pantry/tags/04a2")

    assert response.status_code == 404
