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

`get_cart` / `add_items` / `set_item_quantity` / `remove_item` back the cart
API (CUE-80). Swiggy's `update_cart` *replaces* the cart, so every one of
them is a read-merge-write against the current server cart - written once,
here, so the three mutating routes cannot each invent their own version of
it. See `_user_cart_lock` for the concurrency guarantee, which is
deliberately per-process only.
"""

from __future__ import annotations

import asyncio
import logging
import math
import uuid
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from app.cart.constants import DROPPED_BY_SWIGGY_REASON, MINIMUM_ORDER_VALUE
from app.cart.exceptions import CartItemNotFoundError
from app.cart.schemas import (
    CartItemRequest,
    CartMutationResult,
    ComposeCartResult,
    Ingredient,
    MatchStatus,
    RejectedCartItem,
    SelectedVariant,
)
from app.cart.units import normalize_quantity, parse_pack_size
from app.instamart import service as instamart_service
from app.instamart.exceptions import InstamartDomainError
from app.instamart.schemas import (
    Cart,
    CartItemInput,
    CartLineItem,
    Product,
    ProductVariant,
)
from app.models.cart import CartPlan, CartPlanItem

logger = logging.getLogger(__name__)

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


# --------------------------------------------------------------------------
# Cart API (CUE-80)
# --------------------------------------------------------------------------

# One lock per user, created on demand and dropped once nobody holds or
# awaits it, so a long-lived process doesn't accumulate a lock per user
# who ever touched their cart.
_cart_locks: dict[int, asyncio.Lock] = {}
_cart_lock_users: dict[int, int] = {}
_cart_locks_guard = asyncio.Lock()


@asynccontextmanager
async def _user_cart_lock(user_id: int) -> AsyncIterator[None]:
    """Serialize read-merge-write against one user's cart.

    Read-merge-write is not atomic against Swiggy: two concurrent adds that
    both read the same cart would each write their own merge, and the second
    write would drop the first one's lines. This lock closes that window.

    The guarantee is honest but narrow: it holds **within a single worker
    process only**. A horizontally scaled deployment (or a multi-worker
    uvicorn) can still interleave two requests for the same user across
    processes. Closing that needs a shared lock (Postgres advisory lock or
    Redis); it is deliberately out of scope here, and noted rather than
    papered over.
    """
    async with _cart_locks_guard:
        lock = _cart_locks.setdefault(user_id, asyncio.Lock())
        _cart_lock_users[user_id] = _cart_lock_users.get(user_id, 0) + 1
    try:
        async with lock:
            yield
    finally:
        async with _cart_locks_guard:
            _cart_lock_users[user_id] -= 1
            if _cart_lock_users[user_id] == 0:
                del _cart_lock_users[user_id]
                del _cart_locks[user_id]


def _as_inputs(lines: Iterable[CartLineItem]) -> list[CartItemInput]:
    """Project cart lines back onto `update_cart`'s request shape."""
    return [
        CartItemInput(spin_id=line.spin_id, quantity=line.quantity) for line in lines
    ]


def _merge(
    existing: Iterable[CartItemInput], additions: Iterable[CartItemRequest]
) -> list[CartItemInput]:
    """Merge `additions` onto `existing`, summing quantities per `spin_id`.

    The union, never the delta: `update_cart` replaces the cart, so anything
    omitted here is deleted from the user's real cart. An item already in
    the cart has its quantity *increased* rather than overwritten - the app
    sends "add one more of this", not "make it exactly this many".

    Existing lines keep their order and come first, so a write never
    gratuitously reshuffles the user's cart.
    """
    quantities: dict[str, int] = {item.spin_id: item.quantity for item in existing}
    for addition in additions:
        quantities[addition.spin_id] = (
            quantities.get(addition.spin_id, 0) + addition.quantity
        )
    return [
        CartItemInput(spin_id=spin_id, quantity=quantity)
        for spin_id, quantity in quantities.items()
    ]


def _dropped(
    cart: Cart, requested: Iterable[CartItemRequest], already_rejected: set[str]
) -> list[RejectedCartItem]:
    """Report requested lines missing from the cart Swiggy read back.

    Swiggy does not always fail a write it cannot fully honour: it can
    answer `success: true` and quietly omit an out-of-stock or undeliverable
    line. Diffing the read-back against what we asked for is the only way to
    catch that, and it costs no extra call.
    """
    present = {line.spin_id for line in cart.items}
    return [
        RejectedCartItem(
            spin_id=item.spin_id,
            quantity=item.quantity,
            reason=DROPPED_BY_SWIGGY_REASON,
        )
        for item in requested
        if item.spin_id not in present and item.spin_id not in already_rejected
    ]


async def get_cart(session: AsyncSession, user_id: int) -> Cart:
    """Return the user's current Swiggy server cart (R5.2)."""
    return await instamart_service.get_cart(session, user_id)


async def add_items(
    session: AsyncSession,
    user_id: int,
    *,
    address_id: str,
    items: list[CartItemRequest],
) -> CartMutationResult:
    """Add `items` to the user's cart, preserving whatever it already holds.

    Reads the current cart, merges the additions onto it, and writes the
    union - `update_cart` replaces the cart, so writing only the incoming
    items would silently wipe everything the user already had.

    A `spin_id` Swiggy will not accept is reported per-item; the rest of the
    batch still lands. That is never an error status: a partly-accepted
    batch is a successful call with a non-empty `rejected`.

    Raises:
        InstamartAuthError: The Swiggy link is missing or expired (401).
        InstamartTransportError: Swiggy was unreachable (502).
    """
    async with _user_cart_lock(user_id):
        current = await instamart_service.get_cart(session, user_id)
        baseline = _as_inputs(current.items)
        try:
            cart = await instamart_service.update_cart(
                session, user_id, address_id=address_id, items=_merge(baseline, items)
            )
            rejected: list[RejectedCartItem] = []
        except InstamartDomainError as exc:
            # Swiggy failed the batch as a whole and does not say which line
            # caused it, so fall back to finding out - one write per item,
            # only on this path.
            logger.info(
                "update_cart rejected a %s-item batch for user %s (%s); "
                "retrying item by item to identify the offending line(s).",
                len(items),
                user_id,
                exc.detail,
            )
            cart, rejected = await _add_individually(
                session, user_id, address_id=address_id, baseline=baseline, items=items
            )

        rejected += _dropped(cart, items, {item.spin_id for item in rejected})
        rejected_ids = {item.spin_id for item in rejected}
        added = [item for item in items if item.spin_id not in rejected_ids]
        return CartMutationResult(cart=cart, added=added, rejected=rejected)


async def _add_individually(
    session: AsyncSession,
    user_id: int,
    *,
    address_id: str,
    baseline: list[CartItemInput],
    items: list[CartItemRequest],
) -> tuple[Cart, list[RejectedCartItem]]:
    """Add `items` one at a time, keeping the ones Swiggy accepts.

    Each attempt writes the baseline plus everything accepted so far plus one
    candidate, so a rejected line never removes an accepted one. Costs one
    call per item, which is why it only runs after the single batch write has
    already failed.

    If the *baseline itself* is what Swiggy now refuses - an item that went
    out of stock while sitting in the cart - every attempt fails and the whole
    batch reads as rejected. That is a misattribution, but a safe one: nothing
    is written, and the returned cart is the untouched real one.
    """
    accepted: list[CartItemRequest] = []
    rejected: list[RejectedCartItem] = []
    cart: Cart | None = None

    for item in items:
        try:
            cart = await instamart_service.update_cart(
                session,
                user_id,
                address_id=address_id,
                items=_merge(baseline, [*accepted, item]),
            )
        except InstamartDomainError as exc:
            rejected.append(
                RejectedCartItem(
                    spin_id=item.spin_id, quantity=item.quantity, reason=exc.detail
                )
            )
        else:
            accepted.append(item)

    if cart is None:
        # Nothing landed, so no write returned a cart; read the real one
        # rather than inventing an empty one.
        cart = await instamart_service.get_cart(session, user_id)
    return cart, rejected


async def set_item_quantity(
    session: AsyncSession,
    user_id: int,
    *,
    address_id: str,
    spin_id: str,
    quantity: int,
) -> CartMutationResult:
    """Set one existing line's quantity to `quantity` (absolute, not a delta).

    Raises:
        CartItemNotFoundError: The cart holds no line for `spin_id` (404).
        InstamartAuthError: The Swiggy link is missing or expired (401).
        InstamartTransportError: Swiggy was unreachable (502).
    """
    requested = CartItemRequest(spin_id=spin_id, quantity=quantity)
    async with _user_cart_lock(user_id):
        current = await instamart_service.get_cart(session, user_id)
        if not any(line.spin_id == spin_id for line in current.items):
            raise CartItemNotFoundError

        lines = [
            CartItemInput(
                spin_id=line.spin_id,
                quantity=quantity if line.spin_id == spin_id else line.quantity,
            )
            for line in current.items
        ]
        try:
            cart = await instamart_service.update_cart(
                session, user_id, address_id=address_id, items=lines
            )
        except InstamartDomainError as exc:
            # Same per-item contract as add: the client asked about one line,
            # so report that one line rather than failing the request.
            cart = await instamart_service.get_cart(session, user_id)
            return CartMutationResult(
                cart=cart,
                rejected=[
                    RejectedCartItem(
                        spin_id=spin_id, quantity=quantity, reason=exc.detail
                    )
                ],
            )

        rejected = _dropped(cart, [requested], set())
        return CartMutationResult(
            cart=cart,
            added=[] if rejected else [requested],
            rejected=rejected,
        )


async def remove_item(
    session: AsyncSession, user_id: int, *, address_id: str, spin_id: str
) -> CartMutationResult:
    """Remove one line from the cart, leaving every other line untouched.

    Removing the last line writes an empty item list, which is how
    `update_cart`'s replace semantics express an empty cart.

    Raises:
        CartItemNotFoundError: The cart holds no line for `spin_id` (404).
        InstamartAuthError: The Swiggy link is missing or expired (401).
        InstamartTransportError: Swiggy was unreachable (502).
    """
    async with _user_cart_lock(user_id):
        current = await instamart_service.get_cart(session, user_id)
        if not any(line.spin_id == spin_id for line in current.items):
            raise CartItemNotFoundError

        remaining = _as_inputs(
            line for line in current.items if line.spin_id != spin_id
        )
        cart = await instamart_service.update_cart(
            session, user_id, address_id=address_id, items=remaining
        )
        return CartMutationResult(cart=cart)
