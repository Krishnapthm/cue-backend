"""Batch NFC slug resolution and tag-binding storage (CUE-74).

Deterministic and agent-free by construction: nothing in this module imports
`app.agent`, and resolution is search-and-rank, never a model call. A scan is
a physical rhythm - the user taps jars and expects rows instantly - so the
backend is involved exactly once, when they finish, and answers a whole batch
in one request.

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
from app.tags.constants import SEARCH_CONCURRENCY, TagOutcome
from app.tags.exceptions import TagBindingNotFoundError
from app.tags.schemas import (
    TagBindingUpdate,
    TagResolution,
    TagResolveBatchRequest,
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
    concurrently rather than as 10 serial round-trips.

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

    preferred_spin_ids = await _preferred_spin_ids(session, user_id, address_id)

    queries = sorted(
        {slug for tag_uid, slug in slugs.items() if tag_uid not in reusable}
    )
    candidates_by_slug = await _search_slugs(session, user_id, address_id, queries)
    pantry_item_ids = await _pantry_item_ids(session, user_id, set(slugs.values()))

    results: list[TagResolution] = []
    for tap in request.taps:
        slug = slugs[tap.tag_uid]
        cached = reusable.get(tap.tag_uid)
        if cached is not None:
            results.append(_from_binding(cached, tap, TagOutcome.CACHED))
            continue

        ranked = substitution.rank_candidates(
            candidates_by_slug.get(slug, []),
            # An existing binding's pack size is the household's known refill
            # size, so a re-bind at another address stays the same size rather
            # than silently jumping to whatever is cheapest there.
            preferred_pack_size=_bound_pack_size(bindings.get(tap.tag_uid)),
            preferred_spin_ids=preferred_spin_ids,
        )
        if not ranked:
            results.append(_unresolved(tap))
            continue

        binding = await _bind(
            session,
            user_id=user_id,
            tap=tap,
            slug=slug,
            address_id=address_id,
            best=ranked[0],
            pantry_item_id=pantry_item_ids.get(slug),
        )
        results.append(_from_binding(binding, tap, TagOutcome.BOUND))

    await _stamp_used(session, user_id, list(reusable))
    await session.commit()
    return results


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


def _bound_pack_size(binding: TagBinding | None) -> str | None:
    """The pack size a re-bind should sort towards, if the tag had one."""
    return binding.refill_size if binding is not None else None


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
