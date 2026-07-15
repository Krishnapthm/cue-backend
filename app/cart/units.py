"""Pack-size and ingredient-quantity unit math (R4.5, CUE-15).

Only mass (grams) and volume (millilitres) are modeled - the two dimensions
recipe quantities and Instamart pack sizes actually use. Count-based
ingredients (e.g. "2 onions") and unparseable pack-size strings are handled
by the caller: this module returns a normalized quantity or None, never a
guess.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

_PACK_SIZE_PATTERN = re.compile(r"(?P<qty>[\d.]+)\s*(?P<unit>[a-zA-Z]+)")

# Every alias converts to its base dimension (g or ml) via this multiplier.
_UNIT_TO_BASE: dict[str, tuple[str, Decimal]] = {
    "g": ("g", Decimal(1)),
    "gm": ("g", Decimal(1)),
    "gram": ("g", Decimal(1)),
    "grams": ("g", Decimal(1)),
    "kg": ("g", Decimal(1000)),
    "kilogram": ("g", Decimal(1000)),
    "kilograms": ("g", Decimal(1000)),
    "ml": ("ml", Decimal(1)),
    "millilitre": ("ml", Decimal(1)),
    "millilitres": ("ml", Decimal(1)),
    "milliliter": ("ml", Decimal(1)),
    "milliliters": ("ml", Decimal(1)),
    "l": ("ml", Decimal(1000)),
    "litre": ("ml", Decimal(1000)),
    "litres": ("ml", Decimal(1000)),
    "liter": ("ml", Decimal(1000)),
    "liters": ("ml", Decimal(1000)),
}


def normalize_quantity(quantity: Decimal, unit: str) -> tuple[str, Decimal] | None:
    """Convert a (quantity, unit) pair to its base dimension.

    Returns (base_unit, base_quantity), e.g. (Decimal(1), "kg") -> ("g",
    Decimal(1000)). Returns None if `unit` isn't a recognized mass/volume
    unit.
    """
    base = _UNIT_TO_BASE.get(unit.strip().lower())
    if base is None:
        return None
    base_unit, multiplier = base
    return base_unit, quantity * multiplier


def parse_pack_size(pack_size: str) -> tuple[str, Decimal] | None:
    """Parse a free-text pack size (e.g. "500 ml", "1kg") to (base_unit, base_qty).

    Returns None if the string doesn't contain a recognized mass/volume
    quantity - Swiggy's exact pack-size format isn't documented, so this
    degrades gracefully rather than raising on an unrecognized shape.
    """
    match = _PACK_SIZE_PATTERN.search(pack_size)
    if match is None:
        return None
    try:
        qty = Decimal(match.group("qty"))
    except InvalidOperation:
        return None
    return normalize_quantity(qty, match.group("unit"))
