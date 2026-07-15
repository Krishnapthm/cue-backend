"""Variant selection (CUE-15) and cart composition (CUE-16).

`select_variant` picks the single best purchasable variant for one
ingredient from `search_products` candidates, ranked highest priority first:

  1. In stock - a hard filter; an out-of-stock variant is never selected.
  2. Pack-size sanity: a variant whose pack size parses and needs the least
     overage to cover the ingredient's quantity ranks above one that doesn't
     parse or wastes more.
  3. The preference signal (R4.3): among variants already tied on (1) and
     (2), one matching the ingredient's preferred brand ranks first -
     preference never rescues a bad pack size or an out-of-stock item.
  4. Price, ascending, as the final tiebreak.

No candidates in stock at all resolves to `unavailable`, never a raised
exception - an unresolved ingredient is a normal outcome `compose_cart` must
handle, not an error.

`compose_cart` takes the selections for every ingredient in a session and
writes them as a single `CartPlan` (R5.1), enforcing the Rs 99 minimum
(R5.4) and the address-bound-plan invariant (R3.3).
"""

from __future__ import annotations

import math
import uuid
from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from app.cart.constants import MINIMUM_ORDER_VALUE
from app.cart.schemas import ComposeCartResult, Ingredient, MatchStatus, SelectedVariant
from app.cart.units import normalize_quantity, parse_pack_size
from app.instamart import service as instamart_service
from app.instamart.schemas import CartItemInput, Product, ProductVariant
from app.models.cart import CartPlan, CartPlanItem

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


async def compose_cart(
    session: AsyncSession,
    user_id: int,
    chat_session_id: uuid.UUID,
    address_id: str,
    selected_variants: list[SelectedVariant],
) -> ComposeCartResult:
    """Compose the full cart for `chat_session_id` in one write (R5.1).

    Always supersedes any existing live plan for this session first (R3.3):
    a recompose - whether from a fresh selection or a switched delivery
    address - replaces the plan; `CartPlan` is append-only, never mutated in
    place, so the superseded history stays debuggable. Because the caller
    always passes the full, freshly-sourced set of `selected_variants`, an
    address switch can never carry stale items forward - there is no
    "reuse previous items" path.

    Below the Rs 99 minimum (R5.4), the plan is still recorded, but
    `update_cart` is never called: there's nothing checkout-able yet, and
    the result's `shortfall` guides the user to add more before recomposing.
    """
    await _supersede_live_plan(session, chat_session_id)

    plan = CartPlan(session_id=chat_session_id, address_id=address_id)
    session.add(plan)
    await session.flush()

    for variant in selected_variants:
        session.add(
            CartPlanItem(
                plan_id=plan.id,
                ingredient_name=variant.ingredient_name,
                ingredient_qty=variant.ingredient_qty,
                ingredient_unit=variant.ingredient_unit,
                match_status=variant.match_status.value,
                spin_id=variant.spin_id,
                product_name=variant.product_name,
                pack_size=variant.pack_size,
                unit_price=variant.unit_price,
                quantity=variant.quantity,
                selection_reason=variant.selection_reason,
            )
        )
    await session.commit()

    purchasable: list[tuple[str, int, Decimal]] = []
    for variant in selected_variants:
        if (
            variant.spin_id is None
            or variant.quantity is None
            or variant.unit_price is None
        ):
            continue
        purchasable.append((variant.spin_id, variant.quantity, variant.unit_price))

    subtotal = sum(
        (unit_price * quantity for _, quantity, unit_price in purchasable),
        start=Decimal(0),
    )
    if subtotal < MINIMUM_ORDER_VALUE:
        return ComposeCartResult(
            plan_id=plan.id,
            subtotal=subtotal,
            minimum_order_value=MINIMUM_ORDER_VALUE,
            below_minimum=True,
            shortfall=MINIMUM_ORDER_VALUE - subtotal,
        )

    items = [
        CartItemInput(spin_id=spin_id, quantity=quantity)
        for spin_id, quantity, _ in purchasable
    ]
    await instamart_service.update_cart(
        session, user_id, address_id=address_id, items=items
    )
    cart = await instamart_service.get_cart(session, user_id)

    return ComposeCartResult(
        plan_id=plan.id,
        subtotal=subtotal,
        minimum_order_value=MINIMUM_ORDER_VALUE,
        below_minimum=False,
        shortfall=Decimal(0),
        cart=cart,
    )


async def _supersede_live_plan(
    session: AsyncSession, chat_session_id: uuid.UUID
) -> None:
    """Mark the session's current live plan (if any) superseded (R3.3)."""
    stmt = (
        update(CartPlan)
        .where(CartPlan.session_id == chat_session_id, CartPlan.superseded_at.is_(None))
        .values(superseded_at=datetime.now(UTC))
    )
    await session.execute(stmt)
