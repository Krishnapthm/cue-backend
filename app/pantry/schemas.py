"""Pantry request/response schemas (CUE-69)."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.pantry.constants import (
    DEFAULT_LEVEL,
    LEVEL_MAX,
    LEVEL_MIN,
    MAX_NAME_LENGTH,
    PantryCategory,
)

# The 0-3 stock ordinal: 0 = Out, 1 = Low, 2 = Half, 3 = Full. Anything off
# the scale is rejected before it reaches the service.
PantryLevel = Annotated[int, Field(ge=LEVEL_MIN, le=LEVEL_MAX)]

# Whitespace is stripped before `min_length` runs (see `str_strip_whitespace`
# on the request models), so a name of only spaces is rejected, not stored.
PantryName = Annotated[str, Field(min_length=1, max_length=MAX_NAME_LENGTH)]


class PantryItemCreate(BaseModel):
    """Body of `POST /pantry`.

    A repeat add of a name already in the pantry updates that item rather
    than erroring, so this doubles as the "add or overwrite" payload; see
    `app.pantry.service.upsert_item`.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    name: PantryName
    category: PantryCategory
    level: PantryLevel = DEFAULT_LEVEL


class PantryItemUpdate(BaseModel):
    """Body of `PATCH /pantry/{item_id}`.

    Every field is optional and only the ones actually present are written -
    in practice the Pantry screen sends `{"level": n}` on each slider
    release. An empty body is rejected rather than silently doing nothing.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    name: PantryName | None = None
    category: PantryCategory | None = None
    level: PantryLevel | None = None

    @model_validator(mode="after")
    def at_least_one_field(self) -> Self:
        """Reject a patch that changes nothing.

        Covers both an empty body and one that sends every field as `null`.
        With this in place `None` unambiguously means "not being changed",
        so the service needs no separate notion of set-versus-unset.
        """
        if self.name is None and self.category is None and self.level is None:
            raise ValueError("Provide at least one of: name, category, level.")
        return self


class PantryItemResponse(BaseModel):
    """One pantry item, as returned by every `/pantry` endpoint.

    `last_bought_at` is a timestamp, deliberately not a pre-formatted phrase:
    the client renders "Bought 3 days ago" itself, so the relative wording
    can change without a server release.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    category: PantryCategory
    level: PantryLevel
    last_bought_at: datetime | None
