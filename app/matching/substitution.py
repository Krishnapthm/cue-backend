"""Substitution reasoning on out-of-stock (R4.4, CUE-17).

When the preferred variant for an ingredient is out of stock (or was never
matched), `propose_substitute` searches Instamart again and ranks whatever
comes back by closeness to the preferred pack size, so the alternative
offered to the user is never a wild mismatch when a closer option exists.
"""

from __future__ import annotations

from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from app.cart import units as cart_units
from app.instamart import service as instamart_service
from app.instamart.schemas import Product, ProductVariant
from app.matching.schemas import SubstitutionResult

# Sort-key rank for a candidate whose pack-size distance to the preferred
# variant is known (finite) vs. unknown (unparseable/mismatched units).
_DISTANCE_KNOWN = 0
_DISTANCE_UNKNOWN = 1
_UNKNOWN_DISTANCE_PLACEHOLDER = Decimal(0)

# Sort-key rank for a candidate the user has ordered before vs. one they
# haven't. Leads the key, so a go-to variant outranks a closer or cheaper
# stranger - see `rank_candidates`.
_PREFERRED = 0
_NOT_PREFERRED = 1

# One purchasable candidate: the product it belongs to, the variant that is
# actually orderable, and that variant's price (never None by construction).
RankedCandidate = tuple[Product, ProductVariant, Decimal]


def _pack_size_distance(
    preferred_pack_size: str | None, candidate_pack_size: str | None
) -> tuple[int, Decimal]:
    """Rank-sortable pack-size distance between a preferred and candidate size.

    Returns `(0, distance)` when both sizes parse to the same base unit
    (g or ml), so closer candidates sort first. Returns `(1, 0)` - ranked
    after every known-distance candidate - when either side is missing,
    unparseable, or the two use different base dimensions (e.g. g vs ml).
    """
    if preferred_pack_size is None or candidate_pack_size is None:
        return (_DISTANCE_UNKNOWN, _UNKNOWN_DISTANCE_PLACEHOLDER)

    preferred_parsed = cart_units.parse_pack_size(preferred_pack_size)
    candidate_parsed = cart_units.parse_pack_size(candidate_pack_size)
    if preferred_parsed is None or candidate_parsed is None:
        return (_DISTANCE_UNKNOWN, _UNKNOWN_DISTANCE_PLACEHOLDER)

    preferred_unit, preferred_qty = preferred_parsed
    candidate_unit, candidate_qty = candidate_parsed
    if preferred_unit != candidate_unit:
        return (_DISTANCE_UNKNOWN, _UNKNOWN_DISTANCE_PLACEHOLDER)

    return (_DISTANCE_KNOWN, abs(candidate_qty - preferred_qty))


def _sort_key(
    candidate: RankedCandidate,
    preferred_pack_size: str | None,
    preferred_spin_ids: frozenset[str],
) -> tuple[int, int, Decimal, Decimal, str]:
    """Full deterministic ranking key: go-to, pack size, price, spin_id.

    Candidates the user has ordered before rank first, then by pack-size
    closeness to `preferred_pack_size`, then by lowest price, then by
    `spin_id` ascending as an always-present tie-break so identically-ranked
    candidates resolve the same way every call.
    """
    _, variant, price = candidate
    distance_rank, distance = _pack_size_distance(
        preferred_pack_size, variant.pack_size
    )
    preference_rank = (
        _PREFERRED if variant.spin_id in preferred_spin_ids else _NOT_PREFERRED
    )
    return (preference_rank, distance_rank, distance, price, variant.spin_id)


def rank_candidates(
    products: list[Product],
    *,
    preferred_pack_size: str | None = None,
    preferred_spin_ids: frozenset[str] = frozenset(),
) -> list[RankedCandidate]:
    """Rank search results into a deterministic best-first order.

    The single ranker for every "which variant do we buy?" decision, so
    substitution (R4.4) and NFC tag binding (CUE-74) can never drift into
    disagreeing about what the best variant for a search term is. Only
    purchasable candidates - in stock, with a price - are considered; an
    unpriced or out-of-stock variant is not orderable and is dropped rather
    than ranked last.

    Args:
        products: Whatever `search_products` returned for the term.
        preferred_pack_size: The pack size to sort towards, e.g. "500 g".
            `None` (nothing known to prefer) leaves pack size out of the
            ordering entirely, since every candidate then ranks the same.
        preferred_spin_ids: Variants the user has ordered before, from
            `your_go_to_items`. An empty set - a household with no history -
            simply promotes nothing.

    Returns:
        Every purchasable candidate, best first. Empty when nothing in
        `products` is orderable.
    """
    candidates: list[RankedCandidate] = [
        (product, variant, variant.price)
        for product in products
        for variant in product.variants
        if variant.in_stock is True and variant.price is not None
    ]
    candidates.sort(
        key=lambda candidate: _sort_key(
            candidate, preferred_pack_size, preferred_spin_ids
        )
    )
    return candidates


def purchasable_candidates(products: list[Product]) -> list[RankedCandidate]:
    """Flatten search results to the candidates that can actually be bought.

    Purchasable means in stock and priced: a variant missing either is not
    orderable, so it is dropped rather than ranked last. Order is Swiggy's
    own - products in the order it returned them, variants in the order they
    appeared on each product - which is the order
    `select_preferred_candidate` considers them in, and therefore the order
    the "Not what you wanted?" picker offers them in (CUE-79).

    Args:
        products: Whatever `search_products` returned for the term.

    Returns:
        Every purchasable candidate, in Swiggy's own order. Empty when
        nothing in `products` is orderable.
    """
    return [
        (product, variant, variant.price)
        for product in products
        for variant in product.variants
        if variant.in_stock is True and variant.price is not None
    ]


def prefer_by_brand(
    purchasable: list[RankedCandidate],
    *,
    preferred_brands: frozenset[str] = frozenset(),
) -> RankedCandidate | None:
    """Pick the winner out of an already-collected purchasable list.

    Split out of `select_preferred_candidate` so a caller that also needs the
    full candidate list (CUE-79's single-tap picker) can get both from one
    traversal without re-deriving either. The choice itself is made here and
    nowhere else: two rankers that could disagree about what a slug resolves
    to is exactly the drift this codebase is most exposed to.

    Args:
        purchasable: The output of `purchasable_candidates`.
        preferred_brands: Brands the user has ordered before, from
            `your_go_to_items`.

    Returns:
        The winning candidate, or `None` if `purchasable` is empty.
    """
    if not purchasable:
        return None

    if preferred_brands:
        for candidate in purchasable:
            product, _, _ = candidate
            if product.brand and product.brand in preferred_brands:
                return candidate

    return purchasable[0]


def select_preferred_candidate(
    products: list[Product],
    *,
    preferred_brands: frozenset[str] = frozenset(),
) -> RankedCandidate | None:
    """Pick one purchasable candidate for a tag-bound slug (CUE-74).

    Default is Swiggy's own relevance order: the first purchasable variant
    of the first product `search_products` returned, exactly as Swiggy
    ranked it. If any purchasable product's brand is one the user has
    ordered before (from `your_go_to_items`), that candidate wins instead -
    a search for "sugar" that comes back with four brands should still
    resolve to the one the household actually buys, not whichever brand
    Swiggy's search happens to rank first.

    Args:
        products: Whatever `search_products` returned for the slug, in
            Swiggy's own order.
        preferred_brands: Brands the user has ordered before, from
            `your_go_to_items`. Empty - a household with no order history,
            or a failed go-to lookup - simply defers to the first result.

    Returns:
        The winning candidate, or `None` if nothing in `products` is
        purchasable (in stock, with a price).
    """
    return prefer_by_brand(
        purchasable_candidates(products), preferred_brands=preferred_brands
    )


def _describe_substitution(
    ingredient_name: str,
    preferred_pack_size: str | None,
    product_name: str,
    pack_size: str | None,
) -> str:
    """Build a legible, user-facing reason for the substitution.

    Omits the preferred/candidate pack-size clauses entirely when unknown,
    rather than rendering the literal word "None".
    """
    preferred_clause = f"in {preferred_pack_size} " if preferred_pack_size else ""
    candidate_clause = f" ({pack_size})" if pack_size else ""
    return (
        f"{ingredient_name} {preferred_clause}was out of stock, so we substituted "
        f"{product_name}{candidate_clause} as the closest available option."
    )


async def propose_substitute(
    session: AsyncSession,
    user_id: int,
    address_id: str,
    ingredient_name: str,
    preferred_pack_size: str | None,
    preferred_quantity: int,
) -> SubstitutionResult | None:
    """Propose an in-stock alternative when the preferred variant is unavailable.

    Searches `app.instamart.service.search_products` for `ingredient_name` at
    `address_id`, ranks every purchasable candidate (in stock and priced) by
    closeness to `preferred_pack_size`, then by lowest price, then by
    `spin_id` as a deterministic tie-break, and returns the best match with a
    human-readable reason.

    Args:
        session: DB session, passed through to `search_products` (used there
            to resolve the user's Swiggy access token).
        user_id: The user whose linked Swiggy account is searched.
        address_id: The delivery address search results are scoped to.
        ingredient_name: The ingredient to search for, and the term used in
            the returned reason.
        preferred_pack_size: The pack size that was preferred but is
            unavailable, e.g. "500 g". May be `None` if the preferred item
            had no known pack size.
        preferred_quantity: How many packs were wanted; carried onto the
            substitution unchanged since only the variant is swapped.

    Returns:
        The best-ranked `SubstitutionResult`, or `None` if nothing
        purchasable (in stock, with a price) turned up - the caller should
        then mark the ingredient `match_status='unavailable'`.
    """
    products = await instamart_service.search_products(
        session, user_id, address_id=address_id, query=ingredient_name
    )

    candidates = rank_candidates(products, preferred_pack_size=preferred_pack_size)
    if not candidates:
        return None

    best_product, best_variant, best_price = candidates[0]

    product_name = best_product.name or best_product.brand or ingredient_name

    return SubstitutionResult(
        spin_id=best_variant.spin_id,
        product_name=product_name,
        pack_size=best_variant.pack_size,
        unit_price=best_price,
        image_url=best_variant.image_url,
        rating=best_variant.rating,
        quantity=preferred_quantity,
        reason=_describe_substitution(
            ingredient_name, preferred_pack_size, product_name, best_variant.pack_size
        ),
    )
