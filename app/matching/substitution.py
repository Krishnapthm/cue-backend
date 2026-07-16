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
    candidate: tuple[Product, ProductVariant, Decimal], preferred_pack_size: str | None
) -> tuple[int, Decimal, Decimal, str]:
    """Full deterministic ranking key: pack-size closeness, price, spin_id.

    Candidates rank by pack-size closeness to `preferred_pack_size` first,
    then by lowest price, then by `spin_id` ascending as an always-present
    tie-break so identically-ranked candidates resolve the same way every
    call.
    """
    _, variant, price = candidate
    distance_rank, distance = _pack_size_distance(
        preferred_pack_size, variant.pack_size
    )
    return (distance_rank, distance, price, variant.spin_id)


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

    candidates: list[tuple[Product, ProductVariant, Decimal]] = [
        (product, variant, variant.price)
        for product in products
        for variant in product.variants
        if variant.in_stock is True and variant.price is not None
    ]
    if not candidates:
        return None

    candidates.sort(key=lambda candidate: _sort_key(candidate, preferred_pack_size))
    best_product, best_variant, best_price = candidates[0]

    product_name = best_product.name or best_product.brand or ingredient_name

    return SubstitutionResult(
        spin_id=best_variant.spin_id,
        product_name=product_name,
        pack_size=best_variant.pack_size,
        unit_price=best_price,
        quantity=preferred_quantity,
        reason=_describe_substitution(
            ingredient_name, preferred_pack_size, product_name, best_variant.pack_size
        ),
    )
