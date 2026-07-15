from __future__ import annotations

from decimal import Decimal

from app.cart.schemas import Ingredient, MatchStatus
from app.cart.service import select_variant
from app.instamart.schemas import Product, ProductVariant


def _product(
    brand: str, spin_id: str, pack_size: str, price: str, *, in_stock: bool = True
) -> Product:
    return Product(
        name=f"{brand} product",
        brand=brand,
        variants=[
            ProductVariant(
                spin_id=spin_id,
                pack_size=pack_size,
                price=Decimal(price),
                in_stock=in_stock,
            )
        ],
    )


def test_select_variant_skips_out_of_stock_variants() -> None:
    ingredient = Ingredient(name="milk", quantity=Decimal(200), unit="ml")
    candidates = [
        _product("BrandA", "spin-oos", "200 ml", "20", in_stock=False),
        _product("BrandB", "spin-live", "200 ml", "22"),
    ]

    result = select_variant(ingredient, candidates)

    assert result.spin_id == "spin-live"
    assert result.match_status == MatchStatus.MATCHED


def test_select_variant_returns_unavailable_when_nothing_is_in_stock() -> None:
    ingredient = Ingredient(name="milk", quantity=Decimal(200), unit="ml")
    candidates = [_product("BrandA", "spin-1", "200 ml", "20", in_stock=False)]

    result = select_variant(ingredient, candidates)

    assert result.match_status == MatchStatus.UNAVAILABLE
    assert result.spin_id is None
    assert result.quantity is None
    assert "No in-stock variant" in result.selection_reason


def test_select_variant_prefers_the_pack_size_with_least_overage() -> None:
    ingredient = Ingredient(name="sugar", quantity=Decimal(200), unit="g")
    candidates = [
        _product("BrandA", "spin-500g", "500 g", "50"),
        _product("BrandB", "spin-1kg", "1 kg", "90"),
    ]

    result = select_variant(ingredient, candidates)

    assert result.spin_id == "spin-500g"
    assert result.quantity == 1
    assert result.overage == Decimal(300)
    assert "leaves" in result.selection_reason


def test_select_variant_computes_multi_pack_counts_for_large_needs() -> None:
    ingredient = Ingredient(name="rice", quantity=Decimal(2), unit="kg")
    candidates = [_product("BrandA", "spin-500g", "500 g", "50")]

    result = select_variant(ingredient, candidates)

    assert result.quantity == 4
    assert result.overage == Decimal(0)


def test_select_variant_prefers_the_preferred_brand_among_equally_sane_packs() -> None:
    ingredient = Ingredient(
        name="sugar", quantity=Decimal(200), unit="g", preferred_brand="BrandA"
    )
    candidates = [
        _product("BrandA", "spin-a", "500 g", "100"),
        _product("BrandB", "spin-b", "500 g", "50"),
    ]

    result = select_variant(ingredient, candidates)

    assert result.spin_id == "spin-a"
    assert result.match_status == MatchStatus.MATCHED


def test_select_variant_does_not_let_preference_override_pack_size_sanity() -> None:
    ingredient = Ingredient(
        name="sugar", quantity=Decimal(200), unit="g", preferred_brand="BrandA"
    )
    candidates = [
        _product("BrandA", "spin-oversized", "5 kg", "500"),
        _product("BrandB", "spin-close-fit", "220 g", "25"),
    ]

    result = select_variant(ingredient, candidates)

    assert result.spin_id == "spin-close-fit"
    assert result.match_status == MatchStatus.SUBSTITUTED
    assert "BrandA" in result.selection_reason


def test_select_variant_falls_back_to_price_when_sanity_and_preference_tie() -> None:
    ingredient = Ingredient(name="sugar", quantity=Decimal(200), unit="g")
    candidates = [
        _product("BrandA", "spin-expensive", "500 g", "100"),
        _product("BrandB", "spin-cheap", "500 g", "50"),
    ]

    result = select_variant(ingredient, candidates)

    assert result.spin_id == "spin-cheap"


def test_select_variant_defaults_to_one_pack_when_quantity_is_unspecified() -> None:
    ingredient = Ingredient(name="onion")
    candidates = [_product("BrandA", "spin-1", "1 kg bag", "40")]

    result = select_variant(ingredient, candidates)

    assert result.match_status == MatchStatus.MATCHED
    assert result.quantity == 1
    assert result.overage is None


def test_select_variant_defaults_to_one_pack_when_dimensions_mismatch() -> None:
    """Grams needed against a millilitre pack can't be compared; don't guess."""
    ingredient = Ingredient(name="oil", quantity=Decimal(200), unit="g")
    candidates = [_product("BrandA", "spin-1", "1 L", "150")]

    result = select_variant(ingredient, candidates)

    assert result.quantity == 1
    assert result.overage is None


def test_select_variant_marks_matched_when_no_preference_is_given() -> None:
    ingredient = Ingredient(name="sugar", quantity=Decimal(200), unit="g")
    candidates = [_product("BrandA", "spin-a", "500 g", "50")]

    result = select_variant(ingredient, candidates)

    assert result.match_status == MatchStatus.MATCHED
