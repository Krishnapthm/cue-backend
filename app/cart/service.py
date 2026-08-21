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
writes them as a single `CartPlan` (R5.1) and the address-bound-plan
invariant (R3.3). It merges those selections onto whatever the Swiggy cart
already holds rather than replacing it, using the same read-merge-write
primitives as the cart API below, so items added from the pantry screen (or
an earlier chat session) survive a chat recompose. Compose only ever adds or
increases a line - it never removes one; the Rs 99 minimum (R5.4) is
reporting only; see the function docstring for both.

`get_cart` / `add_items` / `set_item_quantity` / `remove_item` / `clear_cart`
back the cart API (CUE-80). Swiggy's `update_cart` *replaces* the cart, so
every one of them but `clear_cart` is a read-merge-write against the current
server cart - written once, here, so the four mutating routes cannot each
invent their own version of it. `clear_cart` is the one exception: it
discards the cart on purpose, so there is nothing to merge onto, and it (plus
`remove_item`'s last-line case) goes through Swiggy's dedicated `clear_cart`
tool rather than an empty `update_cart` write, which Swiggy does not
document and refuses in practice. See `_user_cart_lock` for the concurrency
guarantee, which is deliberately per-process only.
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

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.cart.constants import (
    DROPPED_BY_SWIGGY_REASON,
    EVICTED_BY_SWIGGY_REASON,
    MINIMUM_ORDER_VALUE,
)
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
from app.instamart.exceptions import (
    InstamartCartReviewRequiredError,
    InstamartDomainError,
)
from app.instamart.schemas import (
    Cart,
    CartItemInput,
    CartLineItem,
    Product,
    ProductVariant,
)
from app.models.cart import CartPlan, CartPlanItem
from app.models.chat import ChatSession
from app.models.tag import TagBinding

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
        image_url=variant.image_url,
        rating=variant.rating,
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
    """Compose this turn's selections into the cart in one write (R5.1).

    Always supersedes any existing live *plan* for this session first (R3.3):
    a recompose - whether from a fresh selection or a switched delivery
    address - replaces the plan bookkeeping row; `CartPlan` is append-only,
    never mutated in place, so the superseded history stays debuggable. This
    never touches the Swiggy cart itself: `selected_variants` is merged onto
    whatever the server cart already holds, under the same lock and `_merge`
    that `add_items` uses (CUE-80), so a chat recompose can never drop a line
    added from the pantry screen or an earlier session. Compose only ever
    adds or increases a line; removing one is always an explicit user action
    (`remove_item`, `clear_cart`), never something a recompose decides.

    The Rs 99 minimum (R5.4) no longer gates the write - a merge can
    legitimately clear it on the strength of what the cart already held, even
    when this turn's own addition alone would not. `update_cart` is skipped
    only when this turn has nothing purchasable to add; `below_minimum` and
    `shortfall` are computed from the real post-write cart and are reporting
    only, guiding the user to add more before checkout.
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

    purchasable = [
        CartItemRequest(spin_id=variant.spin_id, quantity=variant.quantity)
        for variant in selected_variants
        if (
            variant.spin_id is not None
            and variant.quantity is not None
            and variant.unit_price is not None
        )
    ]

    async with _user_cart_lock(user_id):
        if purchasable:
            current = await instamart_service.get_cart(session, user_id)
            baseline_ids = {line.spin_id for line in current.items}
            try:
                cart = await instamart_service.update_cart(
                    session,
                    user_id,
                    address_id=address_id,
                    items=_merge(_as_inputs(current.items), purchasable),
                )
            except InstamartCartReviewRequiredError as exc:
                # The write succeeded with stock-adjusted quantities. The cart
                # read below is the source of truth the report renders, and
                # the graph ends at that report - never checkout - so the
                # user can review it first.
                logger.info(
                    "Instamart adjusted cart quantities for session %s: %s",
                    chat_session_id,
                    exc.detail,
                )
                cart = await instamart_service.get_cart(session, user_id)
            # This merge write can evict a line that was already in the cart
            # and had nothing to do with this turn - observed live on a
            # 17-line cart, where adding one item silently dropped two
            # unrelated ones. `report_cart` already catches a dropped *new*
            # selection by checking `cart.items` directly; a dropped
            # pre-existing line has no representation in this turn's matches
            # at all, so it can only be caught here, against the baseline.
            evicted = baseline_ids - {line.spin_id for line in cart.items}
            if evicted:
                logger.warning(
                    "compose_cart for session %s evicted %d pre-existing "
                    "line(s) it never touched: %s",
                    chat_session_id,
                    len(evicted),
                    sorted(evicted),
                )
        else:
            cart = await instamart_service.get_cart(session, user_id)

    subtotal = sum(
        (line.price * line.quantity for line in cart.items if line.price is not None),
        start=Decimal(0),
    )
    below_minimum = subtotal < MINIMUM_ORDER_VALUE

    return ComposeCartResult(
        plan_id=plan.id,
        subtotal=subtotal,
        minimum_order_value=MINIMUM_ORDER_VALUE,
        below_minimum=below_minimum,
        shortfall=max(MINIMUM_ORDER_VALUE - subtotal, Decimal(0)),
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
    cart: Cart,
    requested: Iterable[CartItemRequest | CartItemInput],
    already_rejected: set[str],
    *,
    reason: str = DROPPED_BY_SWIGGY_REASON,
) -> list[RejectedCartItem]:
    """Report requested lines missing from the cart Swiggy read back.

    Swiggy does not always fail a write it cannot fully honour: it can
    answer `success: true` and quietly omit an out-of-stock or undeliverable
    line - or, on a merge write, quietly drop a line that was already in the
    cart and was never part of this write's request at all (observed on a
    17-line cart: adding one new item came back with two unrelated existing
    lines gone). Diffing the read-back against everything the write was
    supposed to leave in the cart - not just what this call added - is the
    only way to catch either case, and it costs no extra call. `reason`
    distinguishes the two for the caller: `requested` here can be this
    call's own additions or the pre-existing baseline it merged onto.
    """
    present = {line.spin_id for line in cart.items}
    return [
        RejectedCartItem(
            spin_id=item.spin_id,
            quantity=item.quantity,
            reason=reason,
        )
        for item in requested
        if item.spin_id not in present and item.spin_id not in already_rejected
    ]


async def _known_names(
    session: AsyncSession, user_id: int, spin_ids: set[str]
) -> dict[str, str]:
    """Look up product names for `spin_ids` in the user's own records.

    Two places already hold a name against a `spin_id`, both written by us:
    the pantry sticker a restock came from, and the cart plan a chat composed.
    The sticker wins when both know a line - it is the name the user chose to
    put on the jar.
    """
    names: dict[str, str] = {}

    plan_items = await session.execute(
        select(CartPlanItem.spin_id, CartPlanItem.product_name)
        .join(CartPlan, CartPlan.id == CartPlanItem.plan_id)
        .join(ChatSession, ChatSession.id == CartPlan.session_id)
        .where(
            ChatSession.user_id == user_id,
            CartPlanItem.spin_id.in_(spin_ids),
            CartPlanItem.product_name.is_not(None),
        )
        # Newest first: a re-composed plan renamed nothing, but a variant that
        # was resolved again more recently is the better description of it.
        .order_by(CartPlanItem.id.desc())
    )
    for spin_id, product_name in plan_items:
        if spin_id is not None:
            names.setdefault(spin_id, product_name)

    bindings = await session.execute(
        select(TagBinding.spin_id, TagBinding.product_name).where(
            TagBinding.user_id == user_id,
            TagBinding.spin_id.in_(spin_ids),
            TagBinding.product_name.is_not(None),
        )
    )
    for spin_id, product_name in bindings:
        names[spin_id] = product_name

    return names


async def _named(session: AsyncSession, user_id: int, cart: Cart) -> Cart:
    """Fill in product names for the lines Swiggy answered with unnamed.

    A cart read is the only view the app has of lines it did not write itself -
    a chat composes its cart server-side - and Swiggy does not reliably name
    the lines it returns. An unnamed line reaches the user as a placeholder
    word where a product should be, in the one screen whose whole job is to
    agree with their real Instamart cart.

    Display only: Swiggy stays the sole authority on which lines are in the
    cart and how many of each. This never adds, drops or re-quantifies a line.
    """
    unnamed = {line.spin_id for line in cart.items if not line.product_name}
    if not unnamed:
        return cart

    names = await _known_names(session, user_id, unnamed)
    if not names:
        return cart

    return cart.model_copy(
        update={
            "items": [
                line
                if line.product_name or line.spin_id not in names
                else line.model_copy(update={"product_name": names[line.spin_id]})
                for line in cart.items
            ]
        }
    )


async def get_cart(session: AsyncSession, user_id: int) -> Cart:
    """Return the user's current Swiggy server cart (R5.2), lines named."""
    cart = await instamart_service.get_cart(session, user_id)
    return await _named(session, user_id, cart)


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
        # The write above submitted `baseline` too (merged into the same
        # `update_cart` call) - a pre-existing line missing from the
        # read-back was evicted by this write just as surely as a rejected
        # new one, and was invisible before this check existed.
        rejected += _dropped(
            cart,
            baseline,
            {item.spin_id for item in rejected},
            reason=EVICTED_BY_SWIGGY_REASON,
        )
        rejected_ids = {item.spin_id for item in rejected}
        added = [item for item in items if item.spin_id not in rejected_ids]
        return CartMutationResult(
            cart=await _named(session, user_id, cart), added=added, rejected=rejected
        )


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
                cart=await _named(session, user_id, cart),
                rejected=[
                    RejectedCartItem(
                        spin_id=spin_id, quantity=quantity, reason=exc.detail
                    )
                ],
            )

        rejected = _dropped(cart, [requested], set())
        return CartMutationResult(
            cart=await _named(session, user_id, cart),
            added=[] if rejected else [requested],
            rejected=rejected,
        )


async def remove_item(
    session: AsyncSession, user_id: int, *, address_id: str, spin_id: str
) -> CartMutationResult:
    """Remove one line from the cart, leaving every other line untouched.

    Removing the last line would naively write an empty item list through
    `update_cart`, but Swiggy does not document (and live testing shows it
    refuses) an empty `items` array there. Swiggy's dedicated `clear_cart`
    tool is the documented way to empty a cart, so the last-line case is
    routed through it instead.

    Raises:
        CartItemNotFoundError: The cart holds no line for `spin_id` (404).
        InstamartAuthError: The Swiggy link is missing or expired (401).
        InstamartTransportError: Swiggy was unreachable (502).
        InstamartDomainError: Swiggy refused the write (e.g. the last-line
            case hitting a `clear_cart` precondition).
    """
    async with _user_cart_lock(user_id):
        current = await instamart_service.get_cart(session, user_id)
        if not any(line.spin_id == spin_id for line in current.items):
            raise CartItemNotFoundError

        remaining = _as_inputs(
            line for line in current.items if line.spin_id != spin_id
        )
        if remaining:
            cart = await instamart_service.update_cart(
                session, user_id, address_id=address_id, items=remaining
            )
        else:
            await instamart_service.clear_cart(session, user_id)
            cart = await instamart_service.get_cart(session, user_id)
        return CartMutationResult(cart=await _named(session, user_id, cart))


async def clear_cart(
    session: AsyncSession, user_id: int, *, address_id: str
) -> CartMutationResult:
    """Remove every line from the cart via Swiggy's dedicated `clear_cart` tool.

    Unlike `add_items`/`set_item_quantity`/`remove_item`, this never reads
    the current cart first: clearing is the one mutation meant to discard
    everything, so there is nothing to merge onto. `address_id` is accepted
    for API consistency with the other mutating routes even though the
    underlying tool needs no address.

    An earlier version of this wrote an empty `items` list through
    `update_cart` - the natural-looking way to express an empty cart, but
    undocumented and refused by Swiggy in practice (a bare 422 with no
    caller-visible reason). `clear_cart` is the documented tool for this.

    Raises:
        InstamartAuthError: The Swiggy link is missing or expired (401).
        InstamartTransportError: Swiggy was unreachable (502).
        InstamartDomainError: Swiggy refused the clear (e.g. a precondition
            like an order already in flight).
    """
    async with _user_cart_lock(user_id):
        await instamart_service.clear_cart(session, user_id)
        cart = await instamart_service.get_cart(session, user_id)
        return CartMutationResult(cart=await _named(session, user_id, cart))
