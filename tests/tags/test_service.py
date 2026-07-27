from __future__ import annotations

from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.instamart.constants import TOOL_SEARCH_PRODUCTS, TOOL_YOUR_GO_TO_ITEMS
from app.instamart.exceptions import InstamartAuthError
from app.models.pantry import PantryItem
from app.models.tag import TagBinding
from app.models.user import User
from app.pantry.constants import PantryCategory
from app.tags import service
from app.tags.constants import TagOutcome
from app.tags.exceptions import TagBindingNotFoundError
from app.tags.schemas import TagResolveBatchRequest, TagTap
from tests.conftest import InstamartToolCallStub
from tests.tags.conftest import (
    ADDRESS_ID,
    go_to_item,
    go_to_result,
    product,
    search_result,
)

MADHUR_SUGAR = product(
    product_id="p-sugar-madhur",
    name="Madhur Pure Sugar",
    spin_id="spin-madhur",
    price="62.00",
    brand="Madhur",
)
CHEAP_SUGAR = product(
    product_id="p-sugar-generic",
    name="Generic Sugar",
    spin_id="spin-generic",
    price="49.00",
    brand="Generic",
)


def batch(*taps: TagTap, address_id: str = ADDRESS_ID) -> TagResolveBatchRequest:
    return TagResolveBatchRequest(address_id=address_id, taps=list(taps))


def tap(tag_uid: str, text: str, quantity: int = 1) -> TagTap:
    return TagTap(tag_uid=tag_uid, text=text, quantity=quantity)


async def test_a_batch_resolves_every_slug_to_an_orderable_variant(
    db_session: AsyncSession, linked_user: User, instamart: InstamartToolCallStub
) -> None:
    instamart.configure_search_query("sugar", search_result(MADHUR_SUGAR))
    instamart.configure_search_query(
        "tea",
        search_result(
            product(
                product_id="p-tea",
                name="Red Label Tea",
                spin_id="spin-tea",
                price="140.00",
                pack_size="500 g",
            )
        ),
    )

    results = await service.resolve_batch(
        db_session,
        linked_user.id,
        batch(tap("uid-sugar", "sugar", quantity=2), tap("uid-tea", "tea")),
    )

    assert [result.tag_uid for result in results] == ["uid-sugar", "uid-tea"]
    assert [result.outcome for result in results] == [
        TagOutcome.BOUND,
        TagOutcome.BOUND,
    ]
    # spin_id is mandatory on a resolved entry: update_cart addresses items by
    # spinId, so an entry without one would not be orderable.
    assert [result.spin_id for result in results] == ["spin-madhur", "spin-tea"]
    assert results[0].product_name == "Madhur Pure Sugar"
    assert results[0].refill_size == "1 kg"
    assert results[0].unit_price == Decimal("62.00")
    assert results[0].in_stock is True
    assert results[0].quantity == 2
    assert results[1].quantity == 1


async def test_a_second_scan_is_served_from_cache_without_searching(
    db_session: AsyncSession, linked_user: User, instamart: InstamartToolCallStub
) -> None:
    instamart.configure_search_query("sugar", search_result(MADHUR_SUGAR))
    first = await service.resolve_batch(
        db_session, linked_user.id, batch(tap("uid-sugar", "sugar"))
    )
    searches_after_first = len(instamart.tool_calls(TOOL_SEARCH_PRODUCTS))

    second = await service.resolve_batch(
        db_session, linked_user.id, batch(tap("uid-sugar", "sugar"))
    )

    assert second[0].outcome is TagOutcome.CACHED
    assert second[0].spin_id == first[0].spin_id
    assert second[0].unit_price == first[0].unit_price
    assert len(instamart.tool_calls(TOOL_SEARCH_PRODUCTS)) == searches_after_first


async def test_a_go_to_brand_outranks_swiggys_own_search_order(
    db_session: AsyncSession, linked_user: User, instamart: InstamartToolCallStub
) -> None:
    # Swiggy's own order puts the cheap generic first; the household has
    # bought Madhur before, so the go-to brand should win anyway.
    instamart.configure_tool_result(
        TOOL_YOUR_GO_TO_ITEMS,
        go_to_result(go_to_item(product_name="Madhur Pure Sugar", brand="Madhur")),
    )
    instamart.configure_search_query("sugar", search_result(CHEAP_SUGAR, MADHUR_SUGAR))

    results = await service.resolve_batch(
        db_session, linked_user.id, batch(tap("uid-sugar", "sugar"))
    )

    assert results[0].spin_id == "spin-madhur"
    assert results[0].unit_price == Decimal("62.00")


async def test_without_a_go_to_match_the_first_search_result_wins(
    db_session: AsyncSession, linked_user: User, instamart: InstamartToolCallStub
) -> None:
    instamart.configure_search_query("sugar", search_result(MADHUR_SUGAR, CHEAP_SUGAR))

    results = await service.resolve_batch(
        db_session, linked_user.id, batch(tap("uid-sugar", "sugar"))
    )

    assert results[0].spin_id == "spin-madhur"


async def test_a_failed_go_to_fetch_still_resolves_the_batch(
    db_session: AsyncSession, linked_user: User, instamart: InstamartToolCallStub
) -> None:
    instamart.configure_tool_status(TOOL_YOUR_GO_TO_ITEMS, 503)
    instamart.configure_search_query("sugar", search_result(MADHUR_SUGAR))

    results = await service.resolve_batch(
        db_session, linked_user.id, batch(tap("uid-sugar", "sugar"))
    )

    assert results[0].outcome is TagOutcome.BOUND
    assert results[0].spin_id == "spin-madhur"


async def test_go_to_items_is_fetched_once_per_batch(
    db_session: AsyncSession, linked_user: User, instamart: InstamartToolCallStub
) -> None:
    taps = [tap(f"uid-{index}", f"slug{index}") for index in range(10)]
    for index in range(10):
        instamart.configure_search_query(
            f"slug{index}",
            search_result(
                product(
                    product_id=f"p-{index}",
                    name=f"Product {index}",
                    spin_id=f"spin-{index}",
                    price="10.00",
                )
            ),
        )

    results = await service.resolve_batch(db_session, linked_user.id, batch(*taps))

    assert all(result.outcome is TagOutcome.BOUND for result in results)
    assert len(instamart.tool_calls(TOOL_YOUR_GO_TO_ITEMS)) == 1
    # 10 slugs, one search each: 11 upstream calls, not 20.
    assert len(instamart.tool_calls(TOOL_SEARCH_PRODUCTS)) == 10


async def test_a_repeated_slug_is_searched_only_once(
    db_session: AsyncSession, linked_user: User, instamart: InstamartToolCallStub
) -> None:
    instamart.configure_search_query("sugar", search_result(MADHUR_SUGAR))

    results = await service.resolve_batch(
        db_session,
        linked_user.id,
        batch(tap("uid-jar-1", "sugar"), tap("uid-jar-2", "Sugar")),
    )

    assert [result.spin_id for result in results] == ["spin-madhur", "spin-madhur"]
    assert len(instamart.tool_calls(TOOL_SEARCH_PRODUCTS)) == 1


async def test_a_slug_swiggy_has_nothing_for_is_unresolved_not_an_error(
    db_session: AsyncSession, linked_user: User, instamart: InstamartToolCallStub
) -> None:
    instamart.configure_search_query("sugar", search_result(MADHUR_SUGAR))
    instamart.configure_search_query("gulkand", search_result())

    results = await service.resolve_batch(
        db_session,
        linked_user.id,
        batch(tap("uid-gulkand", "gulkand"), tap("uid-sugar", "sugar")),
    )

    assert results[0].outcome is TagOutcome.UNRESOLVED
    assert results[0].spin_id is None
    assert results[0].product_name is None
    assert results[0].unit_price is None
    assert results[0].in_stock is None
    assert results[0].quantity == 1
    # The dead slug does not take the rest of the scan down with it.
    assert results[1].outcome is TagOutcome.BOUND

    stored = await db_session.execute(
        select(TagBinding).where(TagBinding.user_id == linked_user.id)
    )
    assert [binding.tag_uid for binding in stored.scalars().all()] == ["uid-sugar"]


async def test_a_slug_whose_search_fails_is_unresolved_not_fatal(
    db_session: AsyncSession, linked_user: User, instamart: InstamartToolCallStub
) -> None:
    instamart.configure_search_query("sugar", search_result(MADHUR_SUGAR))
    instamart.configure_search_query("tea", {"structuredContent": {}}, status_code=503)

    results = await service.resolve_batch(
        db_session,
        linked_user.id,
        batch(tap("uid-tea", "tea"), tap("uid-sugar", "sugar")),
    )

    assert results[0].outcome is TagOutcome.UNRESOLVED
    assert results[1].outcome is TagOutcome.BOUND


async def test_only_purchasable_candidates_are_ever_bound(
    db_session: AsyncSession, linked_user: User, instamart: InstamartToolCallStub
) -> None:
    instamart.configure_search_query(
        "sugar",
        search_result(
            product(
                product_id="p-oos",
                name="Cheapest But Out Of Stock",
                spin_id="spin-oos",
                price="10.00",
                in_stock=False,
            ),
            {
                "productId": "p-unpriced",
                "name": "Unpriced",
                "variants": [{"spinId": "spin-unpriced", "packSize": "1 kg"}],
            },
            MADHUR_SUGAR,
        ),
    )

    results = await service.resolve_batch(
        db_session, linked_user.id, batch(tap("uid-sugar", "sugar"))
    )

    assert results[0].spin_id == "spin-madhur"


async def test_a_binding_from_another_address_is_reresolved_and_rebound(
    db_session: AsyncSession, linked_user: User, instamart: InstamartToolCallStub
) -> None:
    instamart.configure_search_query("sugar", search_result(MADHUR_SUGAR))
    await service.resolve_batch(
        db_session, linked_user.id, batch(tap("uid-sugar", "sugar"))
    )

    # The same sticker, ordering to a different address, where Swiggy carries
    # a different variant. The cached spin_id may not be orderable there, so
    # it must not be returned as-is.
    instamart.configure_search_query(
        "sugar",
        search_result(
            product(
                product_id="p-sugar-other",
                name="Other Depot Sugar",
                spin_id="spin-other",
                price="70.00",
            )
        ),
    )

    results = await service.resolve_batch(
        db_session,
        linked_user.id,
        batch(tap("uid-sugar", "sugar"), address_id="addr-2"),
    )

    assert results[0].outcome is TagOutcome.BOUND
    assert results[0].spin_id == "spin-other"

    binding = await service.get_binding(db_session, linked_user.id, "uid-sugar")
    assert binding.spin_id == "spin-other"
    assert binding.address_id == "addr-2"


async def test_a_rebind_at_a_new_address_defers_to_first_search_result(
    db_session: AsyncSession, linked_user: User, instamart: InstamartToolCallStub
) -> None:
    instamart.configure_search_query(
        "rice",
        search_result(
            product(
                product_id="p-rice-5kg",
                name="India Gate 5kg",
                spin_id="spin-5kg",
                price="500.00",
                pack_size="5 kg",
            )
        ),
    )
    await service.resolve_batch(
        db_session, linked_user.id, batch(tap("uid-rice", "rice"))
    )

    # At the new address, with no go-to brand to match against, resolution
    # takes Swiggy's own first result - not the pack size closest to the
    # household's previous 5 kg refill.
    instamart.configure_search_query(
        "rice",
        search_result(
            product(
                product_id="p-rice-1kg",
                name="India Gate 1kg",
                spin_id="spin-1kg",
                price="120.00",
                pack_size="1 kg",
            ),
            product(
                product_id="p-rice-5kg-b",
                name="India Gate 5kg",
                spin_id="spin-5kg-b",
                price="520.00",
                pack_size="5 kg",
            ),
        ),
    )

    results = await service.resolve_batch(
        db_session, linked_user.id, batch(tap("uid-rice", "rice"), address_id="addr-2")
    )

    assert results[0].spin_id == "spin-1kg"


async def test_bindings_are_strictly_per_user(
    db_session: AsyncSession,
    linked_user: User,
    other_user: User,
    instamart: InstamartToolCallStub,
) -> None:
    instamart.configure_search_query("sugar", search_result(MADHUR_SUGAR))
    await service.resolve_batch(
        db_session, linked_user.id, batch(tap("uid-shared", "sugar"))
    )

    with pytest.raises(TagBindingNotFoundError):
        await service.get_binding(db_session, other_user.id, "uid-shared")


async def test_a_slug_naming_a_pantry_item_links_to_it(
    db_session: AsyncSession, linked_user: User, instamart: InstamartToolCallStub
) -> None:
    item = PantryItem(
        user_id=linked_user.id,
        name="Sugar",
        name_normalized="sugar",
        category=PantryCategory.GRAINS_AND_PULSES.value,
        level=1,
    )
    db_session.add(item)
    await db_session.commit()
    await db_session.refresh(item)
    instamart.configure_search_query("sugar", search_result(MADHUR_SUGAR))

    results = await service.resolve_batch(
        db_session, linked_user.id, batch(tap("uid-sugar", "  SUGAR  "))
    )

    assert results[0].pantry_item_id == item.id


async def test_a_slug_with_no_pantry_item_still_resolves(
    db_session: AsyncSession, linked_user: User, instamart: InstamartToolCallStub
) -> None:
    instamart.configure_search_query("sugar", search_result(MADHUR_SUGAR))

    results = await service.resolve_batch(
        db_session, linked_user.id, batch(tap("uid-sugar", "sugar"))
    )

    assert results[0].outcome is TagOutcome.BOUND
    assert results[0].pantry_item_id is None


async def test_an_unlinked_account_fails_the_whole_batch(
    db_session: AsyncSession, user: User, instamart: InstamartToolCallStub
) -> None:
    # `user` has no Swiggy link at all, so nothing in the batch could resolve.
    with pytest.raises(InstamartAuthError):
        await service.resolve_batch(
            db_session, user.id, batch(tap("uid-sugar", "sugar"))
        )


async def test_the_same_scan_twice_produces_the_same_variant(
    db_session: AsyncSession, linked_user: User, instamart: InstamartToolCallStub
) -> None:
    # No go-to brand to break the tie: both scans should defer to Swiggy's
    # own first search result, deterministically, every time.
    instamart.configure_search_query(
        "salt",
        search_result(
            product(product_id="p-b", name="Salt B", spin_id="spin-b", price="20.00"),
            product(product_id="p-a", name="Salt A", spin_id="spin-a", price="20.00"),
        ),
    )

    first = await service.resolve_batch(
        db_session, linked_user.id, batch(tap("uid-salt-1", "salt"))
    )
    second = await service.resolve_batch(
        db_session, linked_user.id, batch(tap("uid-salt-2", "salt"))
    )

    assert first[0].spin_id == second[0].spin_id == "spin-b"


async def test_a_cache_hit_stamps_last_used_at(
    db_session: AsyncSession, linked_user: User, instamart: InstamartToolCallStub
) -> None:
    instamart.configure_search_query("sugar", search_result(MADHUR_SUGAR))
    await service.resolve_batch(
        db_session, linked_user.id, batch(tap("uid-sugar", "sugar"))
    )
    bound_at = (
        await service.get_binding(db_session, linked_user.id, "uid-sugar")
    ).last_used_at

    await service.resolve_batch(
        db_session, linked_user.id, batch(tap("uid-sugar", "sugar"))
    )

    binding = await service.get_binding(db_session, linked_user.id, "uid-sugar")
    await db_session.refresh(binding)
    assert binding.last_used_at > bound_at
