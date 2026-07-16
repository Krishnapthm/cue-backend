from __future__ import annotations

from collections.abc import Generator

import pytest

from app.orders.service import _reset_tracking_cache


@pytest.fixture(autouse=True)
def reset_tracking_cache() -> Generator[None]:
    """Clear the process-local tracking cache before and after each test.

    The cache is a module-level dict keyed on (user_id, order_id); without
    this, state from one test's poll-floor assertions could leak into the
    next test's timing window.
    """
    _reset_tracking_cache()
    yield
    _reset_tracking_cache()
