"""Single-tap resolution and its alternates list (CUE-79).

The property that matters most here is not any one of these behaviours but
the one asserted by `test_a_single_tap_resolves_to_what_the_batch_resolves_to`:
the two endpoints share `_resolve_tap`, and if they ever stop agreeing about
what a slug means, the app shows one product during the scan and carts a
different one at Add to cart.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.instamart.constants import TOOL_SEARCH_PRODUCTS, TOOL_YOUR_GO_TO_ITEMS
from app.instamart.exceptions import InstamartAuthError
from app.models.user import User
from app.tags import service
from app.tags.constants import MAX_CANDIDATES, TagOutcome
from app.tags.exceptions import TagBindingNotFoundError
from app.tags.schemas import TagResolveBatchRequest, TagResolveRequest, TagTap
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


def one(
    tag_uid: str = "uid-sugar",
    text: str = "sugar",
    *,
    quantity: int = 1,
    address_id: str = ADDRESS_ID,
) -> TagResolveRequest:
    return TagResolveRequest(
        address_id=address_id, tag_uid=tag_uid, text=text, quantity=quantity
    )


async def test_a_single_tap_resolves_to_an_orderable_variant(
    db_session: AsyncSession, linked_user: User, instamart: InstamartToolCallStub
) -> None:
    instamart.configure_search_query("sugar", search_result(MADHUR_SUGAR))

    result = await service.resolve_one(db_session, linked_user.id, one(quantity=2))

    assert result.outcome is TagOutcome.BOUND
    # spin_id is what makes the row orderable: update_cart addresses by spinId.
    assert result.spin_id == "spin-madhur"
    assert result.product_name == "Madhur Pure Sugar"
    assert result.refill_size == "1 kg"
    assert result.unit_price == Decimal("62.00")
    assert result.in_stock is True
    assert result.quantity == 2
    assert len(instamart.tool_calls(TOOL_SEARCH_PRODUCTS)) == 1


async def test_a_second_tap_of_the_same_tag_is_cached_but_still_searches(
    db_session: AsyncSession, linked_user: User, instamart: InstamartToolCallStub
) -> None:
    instamart.configure_search_query("sugar", search_result(MADHUR_SUGAR))
    first = await service.resolve_one(db_session, linked_user.id, one())
    searches_after_first = len(instamart.tool_calls(TOOL_SEARCH_PRODUCTS))

    second = await service.resolve_one(db_session, linked_user.id, one())

    # The binding still answers the resolution itself - no rebind, no
    # `your_go_to_items` - but the search re-runs so the alternates picker is
    # never left with nothing to offer.
    assert second.outcome is TagOutcome.CACHED
    assert second.spin_id == first.spin_id
    assert second.unit_price == first.unit_price
    assert len(instamart.tool_calls(TOOL_SEARCH_PRODUCTS)) == searches_after_first + 1


async def test_a_cache_hit_still_offers_every_candidate_the_search_turns_up(
    db_session: AsyncSession, linked_user: User, instamart: InstamartToolCallStub
) -> None:
    instamart.configure_search_query("sugar", search_result(MADHUR_SUGAR, CHEAP_SUGAR))
    first = await service.resolve_one(db_session, linked_user.id, one())

    second = await service.resolve_one(db_session, linked_user.id, one())

    # The picker is never hidden on a cache hit: every alternate the search
    # turns up is still offered, with the bound variant among them.
    assert second.outcome is TagOutcome.CACHED
    assert first.spin_id in {candidate.spin_id for candidate in second.candidates}
    assert {candidate.spin_id for candidate in second.candidates} == {
        "spin-madhur",
        "spin-generic",
    }


async def test_candidates_carry_the_options_the_winner_beat(
    db_session: AsyncSession, linked_user: User, instamart: InstamartToolCallStub
) -> None:
    instamart.configure_tool_result(
        TOOL_YOUR_GO_TO_ITEMS,
        go_to_result(go_to_item(product_name="Madhur Pure Sugar", brand="Madhur")),
    )
    # Swiggy ranks the generic first; the household's go-to brand wins.
    instamart.configure_search_query("sugar", search_result(CHEAP_SUGAR, MADHUR_SUGAR))

    result = await service.resolve_one(db_session, linked_user.id, one())

    assert result.spin_id == "spin-madhur"
    # The winner is present and identifiable by its spin_id, and the brand it
    # beat is offered as the alternative.
    assert [candidate.spin_id for candidate in result.candidates] == [
        "spin-generic",
        "spin-madhur",
    ]
    loser = result.candidates[0]
    assert loser.product_name == "Generic Sugar"
    assert loser.product_id == "p-sugar-generic"
    assert loser.refill_size == "1 kg"
    assert loser.unit_price == Decimal("49.00")


async def test_candidates_are_capped_but_always_include_the_winner(
    db_session: AsyncSession, linked_user: User, instamart: InstamartToolCallStub
) -> None:
    # Twenty brands, and the one the household actually buys sorts dead last -
    # well below the cap. Truncating naively would show a row whose product is
    # absent from its own picker.
    others = [
        product(
            product_id=f"p-{index}",
            name=f"Sugar {index}",
            spin_id=f"spin-{index}",
            price="50.00",
            brand=f"Brand {index}",
        )
        for index in range(20)
    ]
    instamart.configure_tool_result(
        TOOL_YOUR_GO_TO_ITEMS,
        go_to_result(go_to_item(product_name="Madhur Pure Sugar", brand="Madhur")),
    )
    instamart.configure_search_query("sugar", search_result(*others, MADHUR_SUGAR))

    result = await service.resolve_one(db_session, linked_user.id, one())

    assert result.spin_id == "spin-madhur"
    assert len(result.candidates) == MAX_CANDIDATES
    assert "spin-madhur" in {candidate.spin_id for candidate in result.candidates}
    # Everything ahead of the winner keeps Swiggy's own order.
    assert [candidate.spin_id for candidate in result.candidates[:-1]] == [
        f"spin-{index}" for index in range(MAX_CANDIDATES - 1)
    ]


async def test_only_purchasable_candidates_are_offered(
    db_session: AsyncSession, linked_user: User, instamart: InstamartToolCallStub
) -> None:
    instamart.configure_search_query(
        "sugar",
        search_result(
            product(
                product_id="p-oos",
                name="Out Of Stock",
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

    result = await service.resolve_one(db_session, linked_user.id, one())

    # An unpriced or out-of-stock variant cannot be carted, so offering it in
    # the picker would only produce a dead end.
    assert [candidate.spin_id for candidate in result.candidates] == ["spin-madhur"]


async def test_a_single_tap_resolves_to_what_the_batch_resolves_to(
    db_session: AsyncSession,
    linked_user: User,
    linked_other_user: User,
    instamart: InstamartToolCallStub,
) -> None:
    """The anti-drift guarantee: one ranker, two endpoints, same answer."""
    instamart.configure_tool_result(
        TOOL_YOUR_GO_TO_ITEMS,
        go_to_result(go_to_item(product_name="Madhur Pure Sugar", brand="Madhur")),
    )
    instamart.configure_search_query("sugar", search_result(CHEAP_SUGAR, MADHUR_SUGAR))

    single = await service.resolve_one(db_session, linked_user.id, one())
    # The same tag, address and order history, resolved the other way. A
    # second user keeps the two from sharing a binding, so the batch really
    # re-ranks rather than reading back what the single tap just wrote.
    batched = await service.resolve_batch(
        db_session,
        linked_other_user.id,
        TagResolveBatchRequest(
            address_id=ADDRESS_ID, taps=[TagTap(tag_uid="uid-sugar", text="sugar")]
        ),
    )

    assert single.outcome is TagOutcome.BOUND
    assert batched[0].outcome is TagOutcome.BOUND
    assert single.spin_id == batched[0].spin_id
    assert single.product_name == batched[0].product_name
    assert single.refill_size == batched[0].refill_size
    assert single.unit_price == batched[0].unit_price


async def test_a_slug_swiggy_has_nothing_for_is_unresolved_not_an_error(
    db_session: AsyncSession, linked_user: User, instamart: InstamartToolCallStub
) -> None:
    instamart.configure_search_query("gulkand", search_result())

    result = await service.resolve_one(
        db_session, linked_user.id, one("uid-gulkand", "gulkand")
    )

    assert result.outcome is TagOutcome.UNRESOLVED
    assert result.spin_id is None
    assert result.product_name is None
    assert result.unit_price is None
    assert result.in_stock is None
    assert result.candidates == []
    assert result.quantity == 1

    # Nothing purchasable means nothing to bind: the tap can be retried later.
    with pytest.raises(TagBindingNotFoundError):
        await service.get_binding(db_session, linked_user.id, "uid-gulkand")


async def test_go_to_items_is_not_fetched_once_per_tap(
    db_session: AsyncSession, linked_user: User, instamart: InstamartToolCallStub
) -> None:
    """The one real cost of going per-tap, and the cache that removes it."""
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

    for index in range(10):
        result = await service.resolve_one(
            db_session, linked_user.id, one(f"uid-{index}", f"slug{index}")
        )
        assert result.outcome is TagOutcome.BOUND

    # A 10-jar scan costs 11 upstream calls, not 20.
    assert len(instamart.tool_calls(TOOL_YOUR_GO_TO_ITEMS)) == 1
    assert len(instamart.tool_calls(TOOL_SEARCH_PRODUCTS)) == 10


async def test_the_brand_cache_is_scoped_to_one_user(
    db_session: AsyncSession,
    linked_user: User,
    linked_other_user: User,
    instamart: InstamartToolCallStub,
) -> None:
    instamart.configure_search_query("sugar", search_result(MADHUR_SUGAR))

    await service.resolve_one(db_session, linked_user.id, one())
    await service.resolve_one(db_session, linked_other_user.id, one())

    # One household's go-to brands must never rank another's tap.
    assert len(instamart.tool_calls(TOOL_YOUR_GO_TO_ITEMS)) == 2


async def test_a_failed_go_to_fetch_still_resolves_the_tap(
    db_session: AsyncSession, linked_user: User, instamart: InstamartToolCallStub
) -> None:
    instamart.configure_tool_status(TOOL_YOUR_GO_TO_ITEMS, 503)
    instamart.configure_search_query("sugar", search_result(MADHUR_SUGAR))

    result = await service.resolve_one(db_session, linked_user.id, one())

    assert result.outcome is TagOutcome.BOUND
    assert result.spin_id == "spin-madhur"


async def test_a_failed_go_to_fetch_is_not_cached(
    db_session: AsyncSession, linked_user: User, instamart: InstamartToolCallStub
) -> None:
    """One transient outage must not cost brand preference for a whole scan."""
    instamart.configure_tool_status(TOOL_YOUR_GO_TO_ITEMS, 503)
    instamart.configure_search_query("sugar", search_result(CHEAP_SUGAR, MADHUR_SUGAR))
    first = await service.resolve_one(db_session, linked_user.id, one("uid-a", "sugar"))

    instamart.configure_tool_result(
        TOOL_YOUR_GO_TO_ITEMS,
        go_to_result(go_to_item(product_name="Madhur Pure Sugar", brand="Madhur")),
    )
    second = await service.resolve_one(
        db_session, linked_user.id, one("uid-b", "sugar")
    )

    # The first tap degraded to Swiggy's own first result; the next tap
    # retried and got the household's actual brand.
    assert first.spin_id == "spin-generic"
    assert second.spin_id == "spin-madhur"


async def test_a_tap_at_a_new_address_is_reresolved(
    db_session: AsyncSession, linked_user: User, instamart: InstamartToolCallStub
) -> None:
    instamart.configure_search_query("sugar", search_result(MADHUR_SUGAR))
    await service.resolve_one(db_session, linked_user.id, one())

    instamart.configure_search_query(
        "sugar",
        search_result(
            product(
                product_id="p-other",
                name="Other Depot Sugar",
                spin_id="spin-other",
                price="70.00",
            )
        ),
    )
    result = await service.resolve_one(
        db_session, linked_user.id, one(address_id="addr-2")
    )

    # The cached variant may simply not be orderable at the new address.
    assert result.outcome is TagOutcome.BOUND
    assert result.spin_id == "spin-other"


async def test_a_relabeled_sticker_is_reresolved_not_served_stale(
    db_session: AsyncSession, linked_user: User, instamart: InstamartToolCallStub
) -> None:
    """Physical tags get reused: the same `tag_uid` written over with a new
    slug must not keep answering as whatever it used to be bound to."""
    instamart.configure_search_query("sugar", search_result(MADHUR_SUGAR))
    first = await service.resolve_one(
        db_session, linked_user.id, one("uid-reused", "sugar")
    )
    assert first.spin_id == "spin-madhur"

    instamart.configure_search_query(
        "sooji",
        search_result(
            product(
                product_id="p-sooji",
                name="Bansi Sooji",
                spin_id="spin-sooji",
                price="45.00",
            )
        ),
    )
    second = await service.resolve_one(
        db_session, linked_user.id, one("uid-reused", "sooji")
    )

    assert second.outcome is TagOutcome.BOUND
    assert second.spin_id == "spin-sooji"
    assert second.product_name == "Bansi Sooji"


async def test_bindings_are_strictly_per_user(
    db_session: AsyncSession,
    linked_user: User,
    linked_other_user: User,
    instamart: InstamartToolCallStub,
) -> None:
    instamart.configure_search_query("sugar", search_result(MADHUR_SUGAR))

    await service.resolve_one(db_session, linked_user.id, one("uid-shared", "sugar"))

    with pytest.raises(TagBindingNotFoundError):
        await service.get_binding(db_session, linked_other_user.id, "uid-shared")


async def test_a_tap_resolved_singly_comes_back_cached_from_the_batch(
    db_session: AsyncSession, linked_user: User, instamart: InstamartToolCallStub
) -> None:
    """What makes keeping both endpoints cheap rather than duplicative."""
    instamart.configure_search_query("sugar", search_result(MADHUR_SUGAR))
    single = await service.resolve_one(db_session, linked_user.id, one())
    searches_after_tap = len(instamart.tool_calls(TOOL_SEARCH_PRODUCTS))

    reconciled = await service.resolve_batch(
        db_session,
        linked_user.id,
        TagResolveBatchRequest(
            address_id=ADDRESS_ID, taps=[TagTap(tag_uid="uid-sugar", text="sugar")]
        ),
    )

    assert reconciled[0].outcome is TagOutcome.CACHED
    assert reconciled[0].spin_id == single.spin_id
    assert len(instamart.tool_calls(TOOL_SEARCH_PRODUCTS)) == searches_after_tap


async def test_an_unlinked_account_fails_the_tap(
    db_session: AsyncSession, user: User, instamart: InstamartToolCallStub
) -> None:
    with pytest.raises(InstamartAuthError):
        await service.resolve_one(db_session, user.id, one())
