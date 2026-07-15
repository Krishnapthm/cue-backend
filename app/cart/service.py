"""Variant selection + quantity math (R4.2/R4.3/R4.5, CUE-15 - core IP).

Given `search_products` candidates for one ingredient, picks the single best
purchasable variant and maps the recipe's needed quantity to a purchasable
pack count. Ranking order, highest priority first:

  1. In stock - a hard filter; an out-of-stock variant is never selected.
  2. Pack-size sanity: a variant whose pack size parses and needs the least
     overage to cover the ingredient's quantity ranks above one that doesn't
     parse or wastes more.
  3. The preference signal (R4.3): among variants already tied on (1) and
     (2), one matching the ingredient's preferred brand ranks first -
     preference never rescues a bad pack size or an out-of-stock item.
  4. Price, ascending, as the final tiebreak.

No candidates in stock at all resolves to `unavailable`, never a raised
exception - an unresolved ingredient is a normal outcome the caller (cart
composition, CUE-16) must handle, not an error.
"""

from __future__ import annotations

import math
from decimal import Decimal

from app.cart.schemas import Ingredient, MatchStatus, SelectedVariant
from app.cart.units import normalize_quantity, parse_pack_size
from app.instamart.schemas import Product, ProductVariant

_NO_IN_STOCK_VARIANT_REASON = "No in-stock variant found for this ingredient."
_INFINITY = Decimal("Infinity")


def select_variant(
    ingredient: Ingredient, candidates: list[Product]
) -> SelectedVariant:
    """Pick the single best variant for `ingredient` from `candidates` (R4.2)."""
    in_stock = [
        (product, variant)
        for product in candidates
        for variant in product.variants
        if variant.in_stock
    ]
    if not in_stock:
        return SelectedVariant(
            ingredient_name=ingredient.name,
            ingredient_qty=ingredient.quantity,
            ingredient_unit=ingredient.unit,
            match_status=MatchStatus.UNAVAILABLE,
            selection_reason=_NO_IN_STOCK_VARIANT_REASON,
        )

    product, variant = min(in_stock, key=lambda pair: _rank_key(ingredient, *pair))
    pack_count, overage = _quantity_math(ingredient, variant)
    is_preferred = _is_preferred_brand(ingredient, product)

    if ingredient.preferred_brand is not None and not is_preferred:
        match_status = MatchStatus.SUBSTITUTED
        reason = (
            f"Preferred brand '{ingredient.preferred_brand}' unavailable or not the "
            f"best match; substituted {_describe(product, variant)}."
        )
    else:
        match_status = MatchStatus.MATCHED
        reason = f"Selected {_describe(product, variant)}."

    if overage is not None and overage > 0:
        reason += (
            f" {pack_count} pack(s) leaves {overage.normalize()} over the "
            "amount needed."
        )

    return SelectedVariant(
        ingredient_name=ingredient.name,
        ingredient_qty=ingredient.quantity,
        ingredient_unit=ingredient.unit,
        match_status=match_status,
        spin_id=variant.spin_id,
        product_name=product.name,
        pack_size=variant.pack_size,
        unit_price=variant.price,
        quantity=pack_count,
        overage=overage,
        selection_reason=reason,
    )


def _describe(product: Product, variant: ProductVariant) -> str:
    label = product.brand or product.name or "product"
    return f"{label} {variant.pack_size}".strip() if variant.pack_size else label


def _is_preferred_brand(ingredient: Ingredient, product: Product) -> bool:
    if ingredient.preferred_brand is None or product.brand is None:
        return False
    return product.brand.strip().lower() == ingredient.preferred_brand.strip().lower()


def _quantity_math(
    ingredient: Ingredient, variant: ProductVariant
) -> tuple[int, Decimal | None]:
    """Map the needed quantity to a pack count, and the resulting overage.

    Returns `(1, None)` when quantity math isn't computable (missing
    quantity/unit, an unparseable pack size, or mismatched dimensions e.g.
    grams needed against a millilitre pack) - one pack is always a safe
    default; the caller reads the `None` overage as "unknown", not "zero".
    """
    pack_math = _pack_math(ingredient, variant)
    if pack_math is None:
        return 1, None
    pack_count, _, overage = pack_math
    return pack_count, overage


def _pack_math(
    ingredient: Ingredient, variant: ProductVariant
) -> tuple[int, Decimal, Decimal] | None:
    """Return (pack_count, pack_qty, overage) in the pack's base unit, or None."""
    if ingredient.quantity is None or ingredient.unit is None or not variant.pack_size:
        return None

    needed = normalize_quantity(ingredient.quantity, ingredient.unit)
    pack = parse_pack_size(variant.pack_size)
    if needed is None or pack is None or needed[0] != pack[0] or pack[1] <= 0:
        return None

    _, needed_qty = needed
    _, pack_qty = pack
    pack_count = max(1, math.ceil(needed_qty / pack_qty))
    return pack_count, pack_qty, pack_count * pack_qty - needed_qty


def _rank_key(
    ingredient: Ingredient, product: Product, variant: ProductVariant
) -> tuple[int, Decimal, int, Decimal]:
    """Lower sorts first: (pack-size sanity, overage ratio, preference, price)."""
    pack_math = _pack_math(ingredient, variant)
    if pack_math is None:
        sanity_rank, overage_ratio = 1, _INFINITY
    else:
        pack_count, pack_qty, overage = pack_math
        needed_qty = pack_count * pack_qty - overage
        overage_ratio = overage / needed_qty if needed_qty > 0 else Decimal(0)
        sanity_rank = 0

    preference_rank = 0 if _is_preferred_brand(ingredient, product) else 1
    price = variant.price if variant.price is not None else _INFINITY
    return (sanity_rank, overage_ratio, preference_rank, price)
