"""Short-TTL in-process cache for a scan's go-to brand set (CUE-79).

Per-tap resolution is the one place batching genuinely bought us something:
`resolve_batch` fetches `your_go_to_items` once for a whole scan, so resolving
tap-by-tap would naively fetch it once per tap and take a 10-jar scan from 11
upstream calls to 20. A household's go-to brands do not change while they walk
their kitchen, so the set is held for a few minutes per `(user_id,
address_id)` and the extra calls disappear.

Deliberately in-process and deliberately small: this is an optimization, never
a source of truth. A cold worker, an evicted entry or an expired one just
re-fetches, and a fetch that fails is not cached at all - a transient
`your_go_to_items` outage must not poison the next five minutes of scanning
with an empty brand set.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.tags.constants import (
    GO_TO_BRANDS_CACHE_MAX_ENTRIES,
    GO_TO_BRANDS_TTL_SECONDS,
)

# Cache key: brand preferences are per user, and the go-to list is fetched
# per delivery address, so both belong in the key.
CacheKey = tuple[int, str]


@dataclass
class BrandPreferenceCache:
    """A tiny TTL map of `(user_id, address_id)` -> go-to brand set.

    Keyed on the monotonic clock, so a system clock adjustment mid-scan
    cannot make an entry look arbitrarily old or young.
    """

    ttl_seconds: float = GO_TO_BRANDS_TTL_SECONDS
    max_entries: int = GO_TO_BRANDS_CACHE_MAX_ENTRIES
    # key -> (expires_at, brands)
    _entries: dict[CacheKey, tuple[float, frozenset[str]]] = field(default_factory=dict)

    def get(self, key: CacheKey) -> frozenset[str] | None:
        """Return the cached brand set, or None if absent or expired.

        Returns:
            The brands last fetched for that user and address. `None` means
            "not known", never "no brands" - an empty `frozenset` is a real,
            cacheable answer for a household with no order history.
        """
        entry = self._entries.get(key)
        if entry is None:
            return None
        expires_at, brands = entry
        if expires_at <= time.monotonic():
            del self._entries[key]
            return None
        return brands

    def set(self, key: CacheKey, brands: frozenset[str]) -> None:
        """Cache a *successfully fetched* brand set.

        Callers must not store a fallback from a failed `your_go_to_items`
        call: an empty set is indistinguishable here from a household with no
        history, and caching the failure would silently drop brand preference
        for the rest of the scan.
        """
        self._sweep()
        if len(self._entries) >= self.max_entries and key not in self._entries:
            # Full and every entry still live: drop the nearest to expiry
            # rather than refusing to cache, so a hot key still benefits.
            oldest = min(self._entries, key=lambda k: self._entries[k][0])
            del self._entries[oldest]
        self._entries[key] = (time.monotonic() + self.ttl_seconds, brands)

    def clear(self) -> None:
        """Forget everything. Used by tests to keep runs independent."""
        self._entries.clear()

    def _sweep(self) -> None:
        """Drop expired entries, so an idle key cannot occupy space forever."""
        now = time.monotonic()
        expired = [
            key for key, (expires_at, _) in self._entries.items() if expires_at <= now
        ]
        for key in expired:
            del self._entries[key]


# Process-wide instance. Scoped to the single-tap path only - `resolve_batch`
# already pays for `your_go_to_items` exactly once per request, and CUE-79
# requires its behaviour to stay unchanged.
go_to_brands = BrandPreferenceCache()
