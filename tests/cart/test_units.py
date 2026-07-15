from __future__ import annotations

from decimal import Decimal

import pytest

from app.cart.units import normalize_quantity, parse_pack_size


@pytest.mark.parametrize(
    ("quantity", "unit", "expected"),
    [
        (Decimal(200), "g", ("g", Decimal(200))),
        (Decimal(1), "kg", ("g", Decimal(1000))),
        (Decimal(2), "KG", ("g", Decimal(2000))),
        (Decimal(500), "ml", ("ml", Decimal(500))),
        (Decimal(1), "l", ("ml", Decimal(1000))),
        (Decimal("1.5"), "litre", ("ml", Decimal("1500.0"))),
    ],
)
def test_normalize_quantity_converts_to_base_units(
    quantity: Decimal, unit: str, expected: tuple[str, Decimal]
) -> None:
    assert normalize_quantity(quantity, unit) == expected


def test_normalize_quantity_returns_none_for_an_unrecognized_unit() -> None:
    assert normalize_quantity(Decimal(2), "pcs") is None


@pytest.mark.parametrize(
    ("pack_size", "expected"),
    [
        ("500 ml", ("ml", Decimal(500))),
        ("1kg", ("g", Decimal(1000))),
        ("1 L", ("ml", Decimal(1000))),
        ("200g", ("g", Decimal(200))),
    ],
)
def test_parse_pack_size_extracts_quantity_and_unit(
    pack_size: str, expected: tuple[str, Decimal]
) -> None:
    assert parse_pack_size(pack_size) == expected


@pytest.mark.parametrize("pack_size", ["Pack of 6", "", "assorted"])
def test_parse_pack_size_returns_none_for_unparseable_strings(pack_size: str) -> None:
    assert parse_pack_size(pack_size) is None
