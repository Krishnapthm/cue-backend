"""HTTP surface of the cart endpoints (CUE-80).

The interesting behaviour here is not the status codes - it is that
`update_cart` *replaces* the cart, so every route has to read-merge-write or
it silently deletes the user's items. `FakeInstamart` implements those real
replace semantics, so these tests assert what the user's cart genuinely ends
up holding rather than that a canned payload came back.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.cart import CartPlan, CartPlanItem
from app.models.provider import ProviderLink
from app.models.tag import TagBinding
from app.models.user import User
from tests.cart.conftest import FakeInstamart
from tests.conftest import InstamartToolCallStub

ADDRESS_ID = "addr-1"


def add_body(*items: tuple[str, int]) -> dict[str, object]:
    return {
        "addressId": ADDRESS_ID,
        "items": [{"spinId": spin_id, "quantity": qty} for spin_id, qty in items],
    }


def spin_ids(payload: dict[str, object]) -> list[str]:
    cart = payload["cart"]
    assert isinstance(cart, dict)
    return [item["spinId"] for item in cart["items"]]


# --------------------------------------------------------------------------
# Authentication
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "path", "json_body"),
    [
        ("get", "/cart", None),
        ("post", "/cart/items", add_body(("spin-1", 1))),
        ("patch", "/cart/items/spin-1", {"addressId": ADDRESS_ID, "quantity": 2}),
        ("delete", f"/cart/items/spin-1?addressId={ADDRESS_ID}", None),
        ("delete", f"/cart?addressId={ADDRESS_ID}", None),
    ],
)
async def test_routes_require_authentication(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    json_body: dict[str, object] | None,
) -> None:
    response = await client.request(method, path, json=json_body)

    assert response.status_code == 401


async def test_expired_swiggy_link_returns_401_and_marks_link_expired(
    authed_client: httpx.AsyncClient,
    mock_instamart_tool_call: InstamartToolCallStub,
    db_session: AsyncSession,
    linked_user: User,
) -> None:
    """Consistent with `resolve_batch`: a live 401 from Swiggy flips the link
    to `expired`, so the next call short-circuits instead of re-asking with a
    token we now know is dead."""
    mock_instamart_tool_call.configure(status_code=401)

    response = await authed_client.get("/cart")

    assert response.status_code == 401
    link = (
        await db_session.execute(
            select(ProviderLink).where(ProviderLink.user_id == linked_user.id)
        )
    ).scalar_one()
    await db_session.refresh(link)
    assert link.status == "expired"


# --------------------------------------------------------------------------
# GET /cart
# --------------------------------------------------------------------------


async def test_get_cart_returns_the_server_cart(
    authed_client: httpx.AsyncClient, fake_instamart: FakeInstamart
) -> None:
    fake_instamart.items = {"spin-1": 2}

    response = await authed_client.get("/cart")

    assert response.status_code == 200
    assert response.json()["items"] == [
        {
            "spinId": "spin-1",
            "quantity": 2,
            "price": "100.00",
            "productName": None,
            "imageUrl": None,
            "rating": None,
        }
    ]


async def test_get_cart_preserves_rating_object_from_the_server_cart(
    authed_client: httpx.AsyncClient, fake_instamart: FakeInstamart
) -> None:
    fake_instamart.items = {"spin-1": 2}
    fake_instamart.metadata = {"spin-1": {"rating": {"value": "4.6", "count": "9.8k"}}}

    response = await authed_client.get("/cart")

    assert response.status_code == 200
    assert response.json()["items"][0]["rating"] == {"value": "4.6", "count": "9.8k"}


async def test_get_cart_reads_a_line_swiggy_names_with_display_name(
    authed_client: httpx.AsyncClient, fake_instamart: FakeInstamart
) -> None:
    """Swiggy does not spell the line's name the same way in every payload."""
    fake_instamart.items = {"spin-1": 1}
    fake_instamart.metadata = {"spin-1": {"displayName": "Amul Gold Milk 1 L"}}

    response = await authed_client.get("/cart")

    assert response.status_code == 200
    assert response.json()["items"][0]["productName"] == "Amul Gold Milk 1 L"


async def test_get_cart_names_an_unnamed_line_from_the_chat_that_composed_it(
    authed_client: httpx.AsyncClient,
    fake_instamart: FakeInstamart,
    db_session: AsyncSession,
    cart_plan: CartPlan,
) -> None:
    """A cart the user never wrote from this device still reads as products.

    A chat composes server-side, so the app learns those lines only by reading
    the cart back - and Swiggy can answer without naming them. The plan that
    composed them holds the name.
    """
    db_session.add(
        CartPlanItem(
            plan_id=cart_plan.id,
            ingredient_name="basmati rice",
            match_status="matched",
            spin_id="spin-1",
            product_name="India Gate Basmati Rice 1 kg",
            pack_size="1 kg",
            unit_price=Decimal("120.00"),
            quantity=1,
            selection_reason="Selected India Gate 1 kg.",
        )
    )
    await db_session.commit()
    fake_instamart.items = {"spin-1": 1}

    response = await authed_client.get("/cart")

    assert response.status_code == 200
    assert response.json()["items"][0]["productName"] == "India Gate Basmati Rice 1 kg"


async def test_get_cart_names_an_unnamed_line_from_the_pantry_sticker(
    authed_client: httpx.AsyncClient,
    fake_instamart: FakeInstamart,
    db_session: AsyncSession,
    linked_user: User,
) -> None:
    db_session.add(
        TagBinding(
            user_id=linked_user.id,
            tag_uid="04A2",
            tag_text="haldi",
            spin_id="spin-1",
            product_name="Everest Haldi Powder 200 g",
            address_id=ADDRESS_ID,
        )
    )
    await db_session.commit()
    fake_instamart.items = {"spin-1": 1}

    response = await authed_client.get("/cart")

    assert response.status_code == 200
    assert response.json()["items"][0]["productName"] == "Everest Haldi Powder 200 g"


async def test_get_cart_leaves_a_line_nobody_can_name_alone(
    authed_client: httpx.AsyncClient, fake_instamart: FakeInstamart
) -> None:
    """An item added in the Swiggy app itself, that Swiggy returns unnamed."""
    fake_instamart.items = {"spin-unknown": 1}

    response = await authed_client.get("/cart")

    assert response.status_code == 200
    assert response.json()["items"][0]["productName"] is None


async def test_get_empty_cart_is_not_an_error(
    authed_client: httpx.AsyncClient, fake_instamart: FakeInstamart
) -> None:
    response = await authed_client.get("/cart")

    assert response.status_code == 200
    assert response.json()["items"] == []


# --------------------------------------------------------------------------
# POST /cart/items - the replace-vs-append trap
# --------------------------------------------------------------------------


async def test_add_preserves_existing_cart_contents(
    authed_client: httpx.AsyncClient, fake_instamart: FakeInstamart
) -> None:
    """The whole point of the read-merge-write: `update_cart` replaces, so a
    naive write of only the incoming items would wipe `spin-1`."""
    fake_instamart.items = {"spin-1": 1}

    response = await authed_client.post("/cart/items", json=add_body(("spin-2", 3)))

    assert response.status_code == 200
    assert spin_ids(response.json()) == ["spin-1", "spin-2"]
    assert fake_instamart.items == {"spin-1": 1, "spin-2": 3}


async def test_add_increases_quantity_of_an_item_already_in_the_cart(
    authed_client: httpx.AsyncClient, fake_instamart: FakeInstamart
) -> None:
    fake_instamart.items = {"spin-1": 2}

    response = await authed_client.post("/cart/items", json=add_body(("spin-1", 3)))

    assert response.status_code == 200
    assert fake_instamart.items == {"spin-1": 5}
    assert response.json()["cart"]["items"][0]["quantity"] == 5


async def test_add_returns_the_resulting_cart_and_the_added_lines(
    authed_client: httpx.AsyncClient, fake_instamart: FakeInstamart
) -> None:
    response = await authed_client.post(
        "/cart/items", json=add_body(("spin-1", 1), ("spin-2", 2))
    )

    body = response.json()
    assert response.status_code == 200
    assert spin_ids(body) == ["spin-1", "spin-2"]
    assert body["added"] == [
        {"spinId": "spin-1", "quantity": 1},
        {"spinId": "spin-2", "quantity": 2},
    ]
    assert body["rejected"] == []


async def test_add_writes_one_batch_when_swiggy_accepts_everything(
    authed_client: httpx.AsyncClient, fake_instamart: FakeInstamart
) -> None:
    """The per-item retry is a failure path only; the happy path stays at one
    write, since Swiggy rate-limits per user."""
    await authed_client.post("/cart/items", json=add_body(("spin-1", 1), ("spin-2", 1)))

    assert len(fake_instamart.writes) == 1


async def test_rejected_spin_id_is_reported_per_item_and_the_rest_still_land(
    authed_client: httpx.AsyncClient, fake_instamart: FakeInstamart
) -> None:
    """The scan screen has to say *which* jar did not make it, so a refused
    line is a 200 with a `rejected` entry - never a 5xx over the whole batch."""
    fake_instamart.items = {"spin-0": 1}
    fake_instamart.rejects = {"spin-2"}

    response = await authed_client.post(
        "/cart/items", json=add_body(("spin-1", 1), ("spin-2", 1), ("spin-3", 1))
    )

    body = response.json()
    assert response.status_code == 200
    assert body["added"] == [
        {"spinId": "spin-1", "quantity": 1},
        {"spinId": "spin-3", "quantity": 1},
    ]
    assert body["rejected"] == [
        {
            "spinId": "spin-2",
            "quantity": 1,
            "reason": "spin-2 is out of stock at this address.",
        }
    ]
    assert spin_ids(body) == ["spin-0", "spin-1", "spin-3"]
    assert fake_instamart.items == {"spin-0": 1, "spin-1": 1, "spin-3": 1}


async def test_every_item_rejected_leaves_the_cart_untouched(
    authed_client: httpx.AsyncClient, fake_instamart: FakeInstamart
) -> None:
    fake_instamart.items = {"spin-0": 4}
    fake_instamart.rejects = {"spin-1", "spin-2"}

    response = await authed_client.post(
        "/cart/items", json=add_body(("spin-1", 1), ("spin-2", 1))
    )

    body = response.json()
    assert response.status_code == 200
    assert body["added"] == []
    assert [item["spinId"] for item in body["rejected"]] == ["spin-1", "spin-2"]
    assert fake_instamart.items == {"spin-0": 4}


async def test_item_silently_dropped_by_swiggy_is_reported_as_rejected(
    authed_client: httpx.AsyncClient, fake_instamart: FakeInstamart
) -> None:
    """Swiggy can answer `success: true` and quietly omit a line. Diffing the
    read-back is the only way to notice, and the client must be told."""
    fake_instamart.drops = {"spin-2"}

    response = await authed_client.post(
        "/cart/items", json=add_body(("spin-1", 1), ("spin-2", 1))
    )

    body = response.json()
    assert response.status_code == 200
    assert body["added"] == [{"spinId": "spin-1", "quantity": 1}]
    assert [item["spinId"] for item in body["rejected"]] == ["spin-2"]
    assert spin_ids(body) == ["spin-1"]


async def test_concurrent_adds_for_one_user_do_not_lose_lines(
    authed_client: httpx.AsyncClient, fake_instamart: FakeInstamart
) -> None:
    """Without the per-user lock, both requests read the same empty cart and
    the second write drops the first one's line."""
    responses = await asyncio.gather(
        authed_client.post("/cart/items", json=add_body(("spin-1", 1))),
        authed_client.post("/cart/items", json=add_body(("spin-2", 1))),
    )

    assert [response.status_code for response in responses] == [200, 200]
    assert fake_instamart.items == {"spin-1": 1, "spin-2": 1}


async def test_add_rejects_an_empty_item_list(
    authed_client: httpx.AsyncClient, fake_instamart: FakeInstamart
) -> None:
    response = await authed_client.post(
        "/cart/items", json={"addressId": ADDRESS_ID, "items": []}
    )

    assert response.status_code == 422
    assert fake_instamart.writes == []


async def test_add_rejects_a_non_positive_quantity(
    authed_client: httpx.AsyncClient, fake_instamart: FakeInstamart
) -> None:
    response = await authed_client.post("/cart/items", json=add_body(("spin-1", 0)))

    assert response.status_code == 422
    assert fake_instamart.writes == []


# --------------------------------------------------------------------------
# PATCH /cart/items/{spin_id}
# --------------------------------------------------------------------------


async def test_patch_sets_an_absolute_quantity_and_leaves_other_lines_alone(
    authed_client: httpx.AsyncClient, fake_instamart: FakeInstamart
) -> None:
    fake_instamart.items = {"spin-1": 1, "spin-2": 5}

    response = await authed_client.patch(
        "/cart/items/spin-1", json={"addressId": ADDRESS_ID, "quantity": 4}
    )

    assert response.status_code == 200
    assert fake_instamart.items == {"spin-1": 4, "spin-2": 5}
    assert response.json()["added"] == [{"spinId": "spin-1", "quantity": 4}]


async def test_patch_on_an_item_not_in_the_cart_is_404(
    authed_client: httpx.AsyncClient, fake_instamart: FakeInstamart
) -> None:
    fake_instamart.items = {"spin-1": 1}

    response = await authed_client.patch(
        "/cart/items/spin-9", json={"addressId": ADDRESS_ID, "quantity": 2}
    )

    assert response.status_code == 404
    assert fake_instamart.writes == []


async def test_patch_rejected_by_swiggy_reports_the_line_not_a_5xx(
    authed_client: httpx.AsyncClient, fake_instamart: FakeInstamart
) -> None:
    fake_instamart.items = {"spin-1": 1}
    fake_instamart.rejects = {"spin-1"}

    response = await authed_client.patch(
        "/cart/items/spin-1", json={"addressId": ADDRESS_ID, "quantity": 99}
    )

    body = response.json()
    assert response.status_code == 200
    assert body["rejected"] == [
        {
            "spinId": "spin-1",
            "quantity": 99,
            "reason": "spin-1 is out of stock at this address.",
        }
    ]
    assert fake_instamart.items == {"spin-1": 1}


async def test_patch_rejects_a_zero_quantity(
    authed_client: httpx.AsyncClient, fake_instamart: FakeInstamart
) -> None:
    """Removing a line is DELETE; there is exactly one way to say each thing."""
    fake_instamart.items = {"spin-1": 1}

    response = await authed_client.patch(
        "/cart/items/spin-1", json={"addressId": ADDRESS_ID, "quantity": 0}
    )

    assert response.status_code == 422
    assert fake_instamart.items == {"spin-1": 1}


# --------------------------------------------------------------------------
# DELETE /cart/items/{spin_id}
# --------------------------------------------------------------------------


async def test_delete_removes_only_the_named_line(
    authed_client: httpx.AsyncClient, fake_instamart: FakeInstamart
) -> None:
    fake_instamart.items = {"spin-1": 1, "spin-2": 2}

    response = await authed_client.delete(f"/cart/items/spin-1?addressId={ADDRESS_ID}")

    assert response.status_code == 200
    assert spin_ids(response.json()) == ["spin-2"]
    assert fake_instamart.items == {"spin-2": 2}


async def test_delete_of_the_last_line_empties_the_cart(
    authed_client: httpx.AsyncClient, fake_instamart: FakeInstamart
) -> None:
    fake_instamart.items = {"spin-1": 1}

    response = await authed_client.delete(f"/cart/items/spin-1?addressId={ADDRESS_ID}")

    assert response.status_code == 200
    assert response.json()["cart"]["items"] == []
    assert fake_instamart.items == {}


async def test_delete_of_an_item_not_in_the_cart_is_404(
    authed_client: httpx.AsyncClient, fake_instamart: FakeInstamart
) -> None:
    fake_instamart.items = {"spin-1": 1}

    response = await authed_client.delete(f"/cart/items/spin-9?addressId={ADDRESS_ID}")

    assert response.status_code == 404
    assert fake_instamart.writes == []


async def test_delete_requires_an_address_id(
    authed_client: httpx.AsyncClient, fake_instamart: FakeInstamart
) -> None:
    response = await authed_client.delete("/cart/items/spin-1")

    assert response.status_code == 422


# --------------------------------------------------------------------------
# DELETE /cart
# --------------------------------------------------------------------------


async def test_clear_removes_every_line(
    authed_client: httpx.AsyncClient, fake_instamart: FakeInstamart
) -> None:
    fake_instamart.items = {"spin-1": 1, "spin-2": 2}

    response = await authed_client.delete(f"/cart?addressId={ADDRESS_ID}")

    assert response.status_code == 200
    assert response.json()["cart"]["items"] == []
    assert fake_instamart.items == {}


async def test_clear_does_not_read_the_cart_first(
    authed_client: httpx.AsyncClient, fake_instamart: FakeInstamart
) -> None:
    """Unlike every other mutating route, clearing never merges - it is the
    one write allowed to discard everything, so there is nothing to read
    first."""
    fake_instamart.items = {"spin-1": 1}

    await authed_client.delete(f"/cart?addressId={ADDRESS_ID}")

    assert fake_instamart.writes == [[]]


async def test_clear_of_an_already_empty_cart_is_a_no_op(
    authed_client: httpx.AsyncClient, fake_instamart: FakeInstamart
) -> None:
    response = await authed_client.delete(f"/cart?addressId={ADDRESS_ID}")

    assert response.status_code == 200
    assert response.json()["cart"]["items"] == []


async def test_clear_requires_an_address_id(
    authed_client: httpx.AsyncClient, fake_instamart: FakeInstamart
) -> None:
    response = await authed_client.delete("/cart")

    assert response.status_code == 422
