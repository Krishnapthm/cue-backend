"""The go-to brand TTL cache (CUE-79).

Small enough to test directly, and worth doing: it is process-wide mutable
state on the scan hot path, so an entry that never expires or a map that grows
without bound would be a slow leak rather than a visible failure.
"""

from __future__ import annotations

from app.tags.cache import BrandPreferenceCache


def test_a_stored_set_is_returned_until_it_expires() -> None:
    cache = BrandPreferenceCache(ttl_seconds=60.0)

    cache.set((1, "addr-1"), frozenset({"Madhur"}))

    assert cache.get((1, "addr-1")) == frozenset({"Madhur"})


def test_an_expired_entry_is_a_miss() -> None:
    # Already stale the instant it is written.
    cache = BrandPreferenceCache(ttl_seconds=0.0)

    cache.set((1, "addr-1"), frozenset({"Madhur"}))

    assert cache.get((1, "addr-1")) is None


def test_an_empty_set_is_a_real_answer_not_a_miss() -> None:
    """A household with no order history must not re-fetch on every tap."""
    cache = BrandPreferenceCache(ttl_seconds=60.0)

    cache.set((1, "addr-1"), frozenset())

    assert cache.get((1, "addr-1")) == frozenset()


def test_entries_are_scoped_to_a_user_and_an_address() -> None:
    cache = BrandPreferenceCache(ttl_seconds=60.0)

    cache.set((1, "addr-1"), frozenset({"Madhur"}))

    assert cache.get((2, "addr-1")) is None
    assert cache.get((1, "addr-2")) is None


def test_the_cache_does_not_grow_past_its_ceiling() -> None:
    cache = BrandPreferenceCache(ttl_seconds=60.0, max_entries=3)

    for user_id in range(10):
        cache.set((user_id, "addr-1"), frozenset({f"Brand {user_id}"}))

    assert len(cache._entries) <= 3
    # The most recent write always survives - it is the one being scanned with.
    assert cache.get((9, "addr-1")) == frozenset({"Brand 9"})


def test_expired_entries_are_swept_rather_than_accumulating() -> None:
    cache = BrandPreferenceCache(ttl_seconds=0.0, max_entries=100)

    for user_id in range(10):
        cache.set((user_id, "addr-1"), frozenset())

    # Every write sweeps what expired before it, so nothing piles up.
    assert len(cache._entries) == 1


def test_clear_forgets_everything() -> None:
    cache = BrandPreferenceCache(ttl_seconds=60.0)
    cache.set((1, "addr-1"), frozenset({"Madhur"}))

    cache.clear()

    assert cache.get((1, "addr-1")) is None
