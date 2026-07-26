"""Pantry domain constants (CUE-69).

The category set and the level scale are part of the API contract, not
client-side presentation choices: the Pantry screen groups by category and
relies on the order declared here, and renders `level` as three segment bars
plus a word. Both therefore live server-side, in one place, and are consumed
by the ORM check constraints, the request/response schemas, and the ordering
of `GET /pantry` alike.
"""

from __future__ import annotations

from enum import StrEnum


class PantryCategory(StrEnum):
    """The fixed server-side category set, in display order.

    Declaration order IS the contract: `app.pantry.service.list_items` orders
    results by each member's position here, and the client groups on that
    order rather than re-deriving it. Adding a category means adding a member
    (and a migration to widen `ck_pantry_item_category_allowed`); reordering
    members reorders the Pantry screen.
    """

    GRAINS_AND_PULSES = "Grains & pulses"
    SPICES_AND_MASALAS = "Spices & masalas"
    VEGETABLES_AND_FRUIT = "Vegetables & fruit"
    DAIRY_AND_EGGS = "Dairy & eggs"
    OILS_AND_CONDIMENTS = "Oils & condiments"
    SNACKS_AND_PACKAGED = "Snacks & packaged"


# Display order as a plain tuple, for the ORDER BY CASE in `list_items` and
# for the check constraint in `app.models.pantry`.
CATEGORY_DISPLAY_ORDER: tuple[PantryCategory, ...] = tuple(PantryCategory)

# `level` is a 0-3 ordinal, NOT a percentage: 0 = Out, 1 = Low, 2 = Half,
# 3 = Full. The word for each value is the client's to render; the API only
# ever carries the integer.
LEVEL_MIN = 0
LEVEL_MAX = 3

# A newly added staple is assumed full - you add it to the pantry because you
# just put it in the cupboard.
DEFAULT_LEVEL = LEVEL_MAX

# Column length ceilings, mirrored by the check constraints on the table.
MAX_NAME_LENGTH = 200
