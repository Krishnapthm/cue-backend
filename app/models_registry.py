"""Imports every domain's ORM models so they register on `Base.metadata`.

Alembic autogenerate and the test harness only see tables whose model modules
have been imported. Add one import line per domain as domains are created.
"""

from __future__ import annotations

from app import models  # noqa: F401  (registers all tables on Base.metadata)
from app.pantry import models as pantry_models  # noqa: F401  (pantry_item)

__all__: list[str] = []
