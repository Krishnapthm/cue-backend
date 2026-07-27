"""NFC slug resolution and tag-binding storage (CUE-74, CUE-79).

Deterministic and agent-free by construction: nothing in this module imports
`app.agent`, and resolution is search-and-rank, never a model call.

Two entry points, one resolution path. `resolve_one` answers a single tap
while the user still has the jar in their hand, so the row can show the real
product rather than a prettified slug; `resolve_batch` answers a finished
scan in one request and remains the reconciliation call at Add to cart, where
anything a tap failed to resolve gets a second chance. Both funnel through
`_resolve_tap`, so the two can never disagree about what a slug resolves to -
a second ranking path is the drift this module is most exposed to.

Every read and write is scoped by `user_id`; a tag UID is only ever looked up
together with its owner, so one household's sticker can never resolve against
another's binding.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.instamart import service as instamart_service
from app.instamart.exceptions import (
    InstamartAuthError,
    InstamartDomainError,
    InstamartTransportError,
)
from app.instamart.schemas import Product
from app.matching import substitution
from app.matching.substitution import RankedCandidate
from app.models.pantry import PantryItem
from app.models.tag import TagBinding
from app.pantry import service as pantry_service
from app.providers import service as provider_service
from app.tags.cache import go_to_brands as go_to_brands_cache
from app.tags.constants import MAX_CANDIDATES, SEARCH_CONCURRENCY, TagOutcome
from app.tags.exceptions import TagBindingNotFoundError
from app.tags.schemas import (
    TagBindingUpdate,
    TagCandidate,
    TagResolution,
    TagResolveBatchRequest,
    TagResolveRequest,
    TagResolveResponse,
    TagTap,
)

logger = logging.getLogger(__name__)


async def resolve_batch(
    session: AsyncSession, user_id: int, request: TagResolveBatchRequest
) -> list[TagResolution]:
    """Resolve a finished scan to one orderable Instamart variant per tap.

    Costs one `your_go_to_items` call plus at most one `search_products` per
    *distinct* slug that is not already bound at the requested address - a
    10-slug scan is 11 upstream calls, not 20, and the searches run
    concurrently rather than as 10 serial round-trips. `search_products` is
    the default resolution path for every distinct slug regardless of what
    `your_go_to_items` returns (CUE-74): go-to items only break ties among
    the search results afterward, by brand - they never gate or reorder the
    search itself, so a household with no order history resolves exactly the
    same way, just always against Swiggy's own first result.

    Args:
        session: An active database session.
        user_id: The scanning user; every binding read and written is scoped
            to them.
        request: The address being ordered to, and the whole set of taps.

    Returns:
        One resolution per tap, in request order. A slug Swiggy has nothing
        for comes back `unresolved` rather than raising, so it cannot take
        the rest of the scan down with it.

    Raises:
        InstamartAuthError: If the user's Swiggy link is missing or expired -
            nothing in the batch can resolve, so this is a whole-request
            failure rather than a per-entry outcome.
    """
    address_id = request.address_id
    slugs = {
        tap.tag_uid: pantry_service.normalize_name(tap.text) for tap in request.taps
    }

    bindings = await _load_bindings(session, user_id, list(slugs))
    # Cache validity is address-scoped: a variant found orderable while
    # ordering to one address may not be orderable at another, and Swiggy
    # gives us no way to check that without searching. Treating an
    # address change as a miss keeps a same-address rescan free (no
    # search at all) while never returning a variant that dies at checkout.
    reusable = {
        tag_uid: binding
        for tag_uid, binding in bindings.items()
        if binding.address_id == address_id
    }

    # CUE-74: go-to-item ranking by spin_id is superseded by brand-preference
    # matching against search results, below - search_products now runs by
    # default on the slug alone, and go-to items are only consulted
    # afterward to break ties among what it returns.
    # preferred_spin_ids = await _preferred_spin_ids(session, user_id, address_id)
    preferred_brands = await _preferred_brands(session, user_id, address_id)

    queries = sorted(
        {slug for tag_uid, slug in slugs.items() if tag_uid not in reusable}
    )
    candidates_by_slug = await _search_slugs(session, user_id, address_id, queries)
    pantry_item_ids = await _pantry_item_ids(session, user_id, set(slugs.values()))

    results: list[TagResolution] = []
    for tap in request.taps:
        slug = slugs[tap.tag_uid]
        # Candidates are discarded here on purpose: nothing renders an
        # alternates picker from a batch result, and the batch response shape
        # is unchanged by CUE-79.
        resolution, _ = await _resolve_tap(
            session,
            user_id=user_id,
            tap=tap,
            slug=slug,
            address_id=address_id,
            cached=reusable.get(tap.tag_uid),
            products=candidates_by_slug.get(slug, []),
            preferred_brands=preferred_brands,
            pantry_item_id=pantry_item_ids.get(slug),
        )
        results.append(resolution)

    await _stamp_used(session, user_id, list(reusable))
    await session.commit()
    return results


async def resolve_one(
    session: AsyncSession, user_id: int, request: TagResolveRequest
) -> TagResolveResponse:
    """Resolve a single tapped sticker, immediately, with its alternates.

    The hot path during a scan. The row lands the moment the tag is read and
    fills in when this answers, so the user sees the product they will
    actually be charged for - "Fortune, not Tata" - while the jar is still in
    their hand, rather than discovering it after the screen has closed.

    Always issues one `search_products` for the slug - including on a repeat
    tap of a tag already bound at this address - so the "not what you're
    looking for" picker always has real alternates to offer, never just the
    current pick. The go-to brand set is cached per `(user_id, address_id)`
    for a few minutes (`app.tags.cache`) so a 10-jar scan does not fetch
    `your_go_to_items` ten times.

    Args:
        session: An active database session.
        user_id: The scanning user; the binding read and written is scoped to
            them.
        request: The tapped tag, its slug, the quantity, and the address
            being ordered to.

    Returns:
        The resolution, plus the other purchasable candidates the search
        turned up. `candidates` is populated on every outcome except
        `unresolved`, so the alternates picker is always available. A slug
        Swiggy has nothing for comes back `unresolved` rather than raising.

    Raises:
        InstamartAuthError: If the user's Swiggy link is missing or expired.
    """
    tap = request.as_tap()
    address_id = request.address_id
    slug = pantry_service.normalize_name(tap.text)

    bindings = await _load_bindings(session, user_id, [tap.tag_uid])
    binding = bindings.get(tap.tag_uid)
    # Same address-scoped cache rule as the batch path: a variant orderable
    # at one address may not be orderable at another, so an address change
    # is a miss.
    reusable = binding is not None and binding.address_id == address_id
    cached = binding if reusable else None

    preferred_brands = await _cached_preferred_brands(session, user_id, address_id)
    products = (await _search_slugs(session, user_id, address_id, [slug])).get(slug, [])
    pantry_item_ids = await _pantry_item_ids(session, user_id, {slug})

    resolution, candidates = await _resolve_tap(
        session,
        user_id=user_id,
        tap=tap,
        slug=slug,
        address_id=address_id,
        cached=cached,
        products=products,
        preferred_brands=preferred_brands,
        pantry_item_id=pantry_item_ids.get(slug),
    )
    if cached is not None:
        await _stamp_used(session, user_id, [tap.tag_uid])
    await session.commit()
    return _with_candidates(resolution, candidates)


async def get_binding(session: AsyncSession, user_id: int, tag_uid: str) -> TagBinding:
    """Return one binding, scoped to its owner.

    Args:
        session: An active database session.
        user_id: The user who must own `tag_uid`.
        tag_uid: The tag whose binding to fetch.

    Returns:
        The matching binding.

    Raises:
        TagBindingNotFoundError: If the tag is unbound, or bound by another
            user - both are the same 404.
    """
    stmt = select(TagBinding).where(
        TagBinding.user_id == user_id, TagBinding.tag_uid == tag_uid
    )
    result = await session.execute(stmt)
    binding = result.scalar_one_or_none()
    if binding is None:
        raise TagBindingNotFoundError
    return binding


async def rebind(
    session: AsyncSession, binding: TagBinding, request: TagBindingUpdate
) -> TagBinding:
    """Point an existing binding at a variant the user picked themselves.

    Written as given rather than re-ranked: the user has already chosen, and
    a correction that quietly re-searched would be able to disagree with them.

    Args:
        session: An active database session.
        binding: The caller's binding, as resolved by the route dependency.
        request: The variant to bind to, and the address it is orderable at.

    Returns:
        The updated binding.
    """
    binding.spin_id = request.spin_id
    binding.product_id = request.product_id
    binding.product_name = request.product_name
    binding.refill_size = request.refill_size
    binding.unit_price = request.unit_price
    binding.address_id = request.address_id
    binding.last_used_at = datetime.now(UTC)
    await session.commit()
    await session.refresh(binding)
    return binding


async def unbind(session: AsyncSession, binding: TagBinding) -> None:
    """Forget a binding, so the next scan of that sticker resolves afresh.

    Args:
        session: An active database session.
        binding: The caller's binding, as resolved by the route dependency.
    """
    await session.delete(binding)
    await session.commit()


async def _resolve_tap(
    session: AsyncSession,
    *,
    user_id: int,
    tap: TagTap,
    slug: str,
    address_id: str,
    cached: TagBinding | None,
    products: list[Product],
    preferred_brands: frozenset[str],
    pantry_item_id: int | None,
) -> tuple[TagResolution, list[RankedCandidate]]:
    """Resolve exactly one tap. The single decision point for both endpoints.

    Every "which variant does this slug mean?" answer in the tag surface is
    made here, from `substitution.prefer_by_brand` over
    `substitution.purchasable_candidates`. `resolve_one` and `resolve_batch`
    differ only in how they gather the inputs - bindings, search results,
    brand preferences - never in how they choose, which is what lets the
    single-tap and batch paths be asserted identical.

    Does not commit; the caller owns the transaction boundary.

    Returns:
        The resolution, and the purchasable candidates it was chosen from -
        empty only on an unresolved slug, where nothing was purchasable. The
        winner (the bound variant, on a cache hit) is always among them, even
        when it ranked below `MAX_CANDIDATES`.
    """
    if cached is not None:
        purchasable = substitution.purchasable_candidates(products)
        return _from_binding(cached, tap, TagOutcome.CACHED), _capped_around(
            purchasable, cached.spin_id
        )

    purchasable = substitution.purchasable_candidates(products)
    best = substitution.prefer_by_brand(purchasable, preferred_brands=preferred_brands)
    if best is None:
        return _unresolved(tap), []

    binding = await _bind(
        session,
        user_id=user_id,
        tap=tap,
        slug=slug,
        address_id=address_id,
        best=best,
        pantry_item_id=pantry_item_id,
    )
    return _from_binding(binding, tap, TagOutcome.BOUND), _capped(purchasable, best)


def _capped(
    purchasable: list[RankedCandidate], best: RankedCandidate
) -> list[RankedCandidate]:
    """Trim the alternates to `MAX_CANDIDATES`, always keeping the winner.

    The picker is a short menu and this rides the scan hot path, but a list
    that omitted the very variant the row is showing would be incoherent - so
    a winner that ranked below the cut displaces the last entry instead of
    being dropped. Relative order is Swiggy's own either way, since the
    winner always sits after everything kept ahead of it.
    """
    kept = purchasable[:MAX_CANDIDATES]
    winning_spin_id = best[1].spin_id
    if any(variant.spin_id == winning_spin_id for _, variant, _ in kept):
        return kept
    return [*purchasable[: MAX_CANDIDATES - 1], best]


def _capped_around(
    purchasable: list[RankedCandidate], spin_id: str
) -> list[RankedCandidate]:
    """`_capped`'s counterpart for a cache hit: no `RankedCandidate` for the
    bound variant to hand in, only its `spin_id`.

    Matches it against the fresh search results so the bound variant still
    anchors the capped list; if the catalog has moved on and it is no longer
    in `purchasable`, there is nothing to force in, so the plain cap stands.
    """
    match = next((c for c in purchasable if c[1].spin_id == spin_id), None)
    if match is None:
        return purchasable[:MAX_CANDIDATES]
    return _capped(purchasable, match)


def _with_candidates(
    resolution: TagResolution, candidates: list[RankedCandidate]
) -> TagResolveResponse:
    """Widen a resolution into the single-tap response, alternates attached."""
    return TagResolveResponse(
        **resolution.model_dump(),
        candidates=[
            TagCandidate(
                spin_id=variant.spin_id,
                product_id=product.product_id,
                product_name=product.name or product.brand,
                refill_size=variant.pack_size,
                unit_price=price,
            )
            for product, variant, price in candidates
        ],
    )


async def _cached_preferred_brands(
    session: AsyncSession, user_id: int, address_id: str
) -> frozenset[str]:
    """`_preferred_brands`, reused across the taps of one scan (CUE-79).

    Only the single-tap path uses this: `resolve_batch` already fetches
    `your_go_to_items` exactly once per request, and its behaviour is
    unchanged by CUE-79.

    A *failed* fetch is not cached. `_preferred_brands` degrades a failure to
    an empty set, which is indistinguishable here from a household that
    genuinely has no go-to brands, so caching it would silently drop brand
    preference for the rest of the scan over one transient error. The cost of
    not caching it is a retry on the next tap, which is what we want.
    """
    key = (user_id, address_id)
    cached = go_to_brands_cache.get(key)
    if cached is not None:
        return cached

    try:
        items = await instamart_service.get_go_to_items(session, user_id, address_id)
    except (InstamartTransportError, InstamartDomainError):
        logger.warning(
            "your_go_to_items failed for user %s; deferring to first search result",
            user_id,
            exc_info=True,
        )
        return frozenset()

    brands = frozenset(item.brand for item in items if item.brand)
    go_to_brands_cache.set(key, brands)
    return brands


async def _load_bindings(
    session: AsyncSession, user_id: int, tag_uids: list[str]
) -> dict[str, TagBinding]:
    """Fetch the caller's existing bindings for the scanned tags, keyed by UID.

    One query for the whole batch. The `IN (...)` is bounded by
    `MAX_TAPS_PER_BATCH`, so it cannot grow unboundedly from client input.
    """
    if not tag_uids:
        return {}
    stmt = select(TagBinding).where(
        TagBinding.user_id == user_id, TagBinding.tag_uid.in_(tag_uids)
    )
    result = await session.execute(stmt)
    return {binding.tag_uid: binding for binding in result.scalars().all()}


async def _preferred_spin_ids(
    session: AsyncSession, user_id: int, address_id: str
) -> frozenset[str]:
    """Fetch the user's go-to variants once per batch, as a set of spin ids.

    Unused as of CUE-74 - see the comment in `resolve_batch` - kept in place
    rather than deleted in case spin_id-level ranking is reinstated.

    Ranking input, never a dependency of resolution: a household with no
    order history, or a `your_go_to_items` call that fails, degrades to plain
    pack-size/price ranking rather than failing the scan. An auth failure is
    the one exception - with no usable Swiggy session the searches cannot run
    either, so it is left to propagate.
    """
    try:
        items = await instamart_service.get_go_to_items(session, user_id, address_id)
    except (InstamartTransportError, InstamartDomainError):
        logger.warning(
            "your_go_to_items failed for user %s; ranking on pack size and price only",
            user_id,
            exc_info=True,
        )
        return frozenset()
    preferences = instamart_service.normalize_preferences(items)
    return frozenset(signal.spin_id for signal in preferences.values())


async def _preferred_brands(
    session: AsyncSession, user_id: int, address_id: str
) -> frozenset[str]:
    """Fetch the user's go-to brands once per batch, for post-search matching.

    Search is the default action for every unresolved slug regardless of
    this call's outcome (CUE-74): `your_go_to_items` only narrows what
    `search_products` already returned, it never gates or reorders the
    search itself. A household with no order history, or a `your_go_to_items`
    call that fails, simply defers to Swiggy's own first result via
    `substitution.select_preferred_candidate`. An auth failure is the one
    exception - with no usable Swiggy session the searches cannot run
    either, so it is left to propagate.
    """
    try:
        items = await instamart_service.get_go_to_items(session, user_id, address_id)
    except (InstamartTransportError, InstamartDomainError):
        logger.warning(
            "your_go_to_items failed for user %s; deferring to first search result",
            user_id,
            exc_info=True,
        )
        return frozenset()
    return frozenset(item.brand for item in items if item.brand)


async def _search_slugs(
    session: AsyncSession, user_id: int, address_id: str, queries: list[str]
) -> dict[str, list[Product]]:
    """Search every unbound slug concurrently, bounded by a semaphore.

    The token is resolved once, serially, because a single `AsyncSession`
    cannot be used from several coroutines at a time; the concurrent leg is
    then pure HTTP. A slug whose own search fails yields no candidates - it
    will be reported `unresolved` - rather than failing the batch, except for
    an auth failure, which means no slug can resolve.
    """
    if not queries:
        return {}

    token = await instamart_service.resolve_access_token(session, user_id)
    semaphore = asyncio.Semaphore(SEARCH_CONCURRENCY)

    async def search(query: str) -> list[Product]:
        async with semaphore:
            return await instamart_service.search_products_with_token(
                token, address_id=address_id, query=query
            )

    outcomes = await asyncio.gather(
        *(search(query) for query in queries), return_exceptions=True
    )

    found: dict[str, list[Product]] = {}
    for query, outcome in zip(queries, outcomes, strict=True):
        if isinstance(outcome, InstamartAuthError):
            await provider_service.mark_link_expired(session, user_id)
            raise outcome
        if isinstance(outcome, BaseException):
            logger.warning(
                "search_products failed for slug %r; reporting it unresolved",
                query,
                exc_info=outcome,
            )
            found[query] = []
            continue
        found[query] = outcome
    return found


async def _pantry_item_ids(
    session: AsyncSession, user_id: int, normalized_names: set[str]
) -> dict[str, int]:
    """Map each scanned slug onto the caller's pantry item of that name.

    Matched on `name_normalized` via `app.pantry.service.normalize_name`, so
    "the same staple" means exactly one thing across the codebase. A slug
    with no pantry item simply has no entry - a sticker can name something
    the user never added by hand.
    """
    names = {name for name in normalized_names if name}
    if not names:
        return {}
    stmt = select(PantryItem.name_normalized, PantryItem.id).where(
        PantryItem.user_id == user_id, PantryItem.name_normalized.in_(names)
    )
    result = await session.execute(stmt)
    return {row.name_normalized: row.id for row in result.all()}


async def _bind(
    session: AsyncSession,
    *,
    user_id: int,
    tap: TagTap,
    slug: str,
    address_id: str,
    best: RankedCandidate,
    pantry_item_id: int | None,
) -> TagBinding:
    """Persist the winning variant as this tag's binding.

    A single `INSERT ... ON CONFLICT DO UPDATE` rather than a select-then-
    write, so two devices scanning the same sticker at once cannot race into
    an integrity error. Does not commit - `resolve_batch` owns the
    transaction boundary for the whole scan.
    """
    product, variant, price = best
    values = {
        "user_id": user_id,
        "tag_uid": tap.tag_uid,
        "tag_text": tap.text,
        "spin_id": variant.spin_id,
        "product_id": product.product_id,
        "product_name": product.name or product.brand or slug,
        "refill_size": variant.pack_size,
        "unit_price": price,
        "address_id": address_id,
        "pantry_item_id": pantry_item_id,
        "last_used_at": datetime.now(UTC),
    }
    stmt = (
        pg_insert(TagBinding)
        .values(**values)
        .on_conflict_do_update(
            constraint="uq_tag_binding_user_tag",
            set_={key: value for key, value in values.items() if key != "user_id"},
        )
        .returning(TagBinding)
        # On the conflict branch the row is already in the identity map from
        # `_load_bindings`, and SQLAlchemy will not overwrite a live object's
        # attributes by default - without this, the response would carry the
        # stale pre-update variant even though the UPDATE did happen.
        .execution_options(populate_existing=True)
    )
    result = await session.execute(stmt)
    return result.scalar_one()


async def _stamp_used(session: AsyncSession, user_id: int, tag_uids: list[str]) -> None:
    """Mark the cache-hit tags as used, in one statement. Does not commit."""
    if not tag_uids:
        return
    await session.execute(
        update(TagBinding)
        .where(TagBinding.user_id == user_id, TagBinding.tag_uid.in_(tag_uids))
        .values(last_used_at=datetime.now(UTC))
    )


def _from_binding(
    binding: TagBinding, tap: TagTap, outcome: TagOutcome
) -> TagResolution:
    """Render a stored binding as one entry of the batch response.

    `in_stock` is `True` on every resolved entry because only in-stock,
    priced candidates are ever bound; on a cache hit that reflects stock as
    of bind time, which is the most a cached answer can honestly claim.
    """
    return TagResolution(
        tag_uid=binding.tag_uid,
        text=tap.text,
        outcome=outcome,
        spin_id=binding.spin_id,
        product_id=binding.product_id,
        product_name=binding.product_name,
        refill_size=binding.refill_size,
        unit_price=binding.unit_price,
        in_stock=True,
        pantry_item_id=binding.pantry_item_id,
        quantity=tap.quantity,
    )


def _unresolved(tap: TagTap) -> TagResolution:
    """Render a tap Swiggy had nothing purchasable for."""
    return TagResolution(
        tag_uid=tap.tag_uid,
        text=tap.text,
        outcome=TagOutcome.UNRESOLVED,
        quantity=tap.quantity,
    )
