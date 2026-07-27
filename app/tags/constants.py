"""NFC tag-binding constants (CUE-74).

The column ceilings mirrored by `app.models.tag`'s check constraints, plus the
two numbers that bound what one batch request can cost upstream.
"""

from __future__ import annotations

from enum import StrEnum


class TagOutcome(StrEnum):
    """How one tap in a batch was resolved.

    `UNRESOLVED` is a normal per-entry outcome inside a 200, never an error:
    one slug Swiggy has nothing for must not fail the other nine.
    """

    # Served from an existing binding, valid at the requested address.
    CACHED = "cached"
    # Freshly searched, ranked and persisted on this request.
    BOUND = "bound"
    # Nothing purchasable; every product field is null.
    UNRESOLVED = "unresolved"


# Ceiling on `taps` per request, so a batch cannot fan out unboundedly. Well
# above a realistic kitchen scan, and far below anything that would strain
# Swiggy's advertised future quota of 120 req/min per user.
MAX_TAPS_PER_BATCH = 50

# How many `search_products` calls may be in flight at once. A 10-slug scan
# must not cost 10 serial round-trips, but an unbounded fan-out would burn the
# whole per-minute quota on a single request.
SEARCH_CONCURRENCY = 5

# Ceiling on the alternates offered alongside a single-tap resolution
# (CUE-79). The picker is a short menu, not a catalogue, and the payload sits
# on the scan hot path - but the winner is always among them, even when it
# ranked below the cut.
MAX_CANDIDATES = 10

# How long a user's go-to brand set stays reusable across taps of one scan
# (CUE-79). Per-tap resolution would otherwise fetch `your_go_to_items` once
# per tap, doubling a 10-jar scan's upstream cost; a household's go-to brands
# do not change mid-scan, so a few minutes of staleness is free.
GO_TO_BRANDS_TTL_SECONDS = 300.0

# Ceiling on cached brand sets, so a long-lived process serving many users
# cannot grow the cache without bound. Expired entries are swept on insert;
# this is the backstop when every entry is still live.
GO_TO_BRANDS_CACHE_MAX_ENTRIES = 1024

# Column length ceilings, mirrored by the check constraints on `tag_binding`
# and by the request schemas, so an over-long value is a 422 rather than a
# constraint violation.
MAX_TAG_UID_LENGTH = 100
MAX_TAG_TEXT_LENGTH = 200
MAX_SPIN_ID_LENGTH = 100
MAX_PRODUCT_ID_LENGTH = 100
MAX_PRODUCT_NAME_LENGTH = 300
MAX_REFILL_SIZE_LENGTH = 50
MAX_ADDRESS_ID_LENGTH = 100

# Per-tap quantity bounds. A tap means "one refill of this"; the client may
# coalesce repeat taps of the same sticker into a count.
MIN_TAP_QUANTITY = 1
MAX_TAP_QUANTITY = 99
DEFAULT_TAP_QUANTITY = 1
