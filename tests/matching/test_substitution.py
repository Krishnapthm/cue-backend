from __future__ import annotations

from decimal import Decimal
from typing import Any, cast

from sqlalchemy.ext.asyncio import AsyncSession

from app.instamart import service as instamart_service
from app.instamart.schemas import Product
from app.matching import substitution
from app.matching.schemas import SubstitutionResult

# The mocked search_products ignores the session entirely - no real DB needed.
_SESSION = cast(AsyncSession, object())
_USER_ID = 1
_ADDRESS_ID = "addr-1"
_INGREDIENT = "milk"


def _patch_search_products(monkeypatch: Any, products: list[Product]) -> None:
    async def fake_search_products(
        session: AsyncSession,
        user_id: int,
        *,
        address_id: str,
        query: str,
        offset: int = 0,
    ) -> list[Product]:
        return products

    monkeypatch.setattr(instamart_service, "search_products", fake_search_products)


async def test_returns_substitute_for_out_of_stock_preferred(monkeypatch: Any) -> None:
    products = [
        Product.model_validate(
            {
                "productId": "prod-1",
                "name": "Amul Milk",
                "brand": "Amul",
                "variants": [
                    {
                        "spinId": "spin-oos",
                        "packSize": "500 ml",
                        "price": "27.00",
                        "inStock": False,
                    },
                    {
                        "spinId": "spin-alt",
                        "packSize": "500 ml",
                        "price": "28.00",
                        "inStock": True,
                    },
                ],
            }
        )
    ]
    _patch_search_products(monkeypatch, products)

    result = await substitution.propose_substitute(
        _SESSION,
        _USER_ID,
        _ADDRESS_ID,
        _INGREDIENT,
        preferred_pack_size="500 ml",
        preferred_quantity=2,
    )

    assert result is not None
    assert result.spin_id == "spin-alt"
    assert result.unit_price == Decimal("28.00")
    assert result.pack_size == "500 ml"
    assert result.quantity == 2
    assert result.reason
    assert "None" not in result.reason


async def test_closest_pack_size_wins_among_multiple_candidates(
    monkeypatch: Any,
) -> None:
    products = [
        Product.model_validate(
            {
                "productId": "prod-1",
                "name": "Some Flour",
                "variants": [
                    {"spinId": "spin-1kg", "packSize": "1 kg", "price": "50.00"},
                    {"spinId": "spin-450g", "packSize": "450 g", "price": "40.00"},
                    {"spinId": "spin-2kg", "packSize": "2 kg", "price": "90.00"},
                ],
            }
        )
    ]
    _patch_search_products(monkeypatch, products)

    result = await substitution.propose_substitute(
        _SESSION,
        _USER_ID,
        _ADDRESS_ID,
        "flour",
        preferred_pack_size="500 g",
        preferred_quantity=1,
    )

    assert result is not None
    assert result.spin_id == "spin-450g"


async def test_tie_on_distance_breaks_by_lowest_price(monkeypatch: Any) -> None:
    products = [
        Product.model_validate(
            {
                "productId": "prod-1",
                "name": "Sugar",
                "variants": [
                    # Both 100 g away from the 500 g preferred size.
                    {"spinId": "spin-expensive", "packSize": "600 g", "price": "60.00"},
                    {"spinId": "spin-cheap", "packSize": "400 g", "price": "30.00"},
                ],
            }
        )
    ]
    _patch_search_products(monkeypatch, products)

    result = await substitution.propose_substitute(
        _SESSION,
        _USER_ID,
        _ADDRESS_ID,
        "sugar",
        preferred_pack_size="500 g",
        preferred_quantity=1,
    )

    assert result is not None
    assert result.spin_id == "spin-cheap"


async def test_tie_on_distance_and_price_breaks_by_spin_id(monkeypatch: Any) -> None:
    products = [
        Product.model_validate(
            {
                "productId": "prod-1",
                "name": "Sugar",
                "variants": [
                    {"spinId": "spin-b", "packSize": "500 g", "price": "30.00"},
                    {"spinId": "spin-a", "packSize": "500 g", "price": "30.00"},
                ],
            }
        )
    ]
    _patch_search_products(monkeypatch, products)

    result = await substitution.propose_substitute(
        _SESSION,
        _USER_ID,
        _ADDRESS_ID,
        "sugar",
        preferred_pack_size="500 g",
        preferred_quantity=1,
    )

    assert result is not None
    assert result.spin_id == "spin-a"


async def test_empty_search_results_returns_none(monkeypatch: Any) -> None:
    _patch_search_products(monkeypatch, [])

    result = await substitution.propose_substitute(
        _SESSION,
        _USER_ID,
        _ADDRESS_ID,
        _INGREDIENT,
        preferred_pack_size="500 ml",
        preferred_quantity=1,
    )

    assert result is None


async def test_all_out_of_stock_returns_none(monkeypatch: Any) -> None:
    products = [
        Product.model_validate(
            {
                "productId": "prod-1",
                "name": "Amul Milk",
                "variants": [
                    {
                        "spinId": "spin-1",
                        "packSize": "500 ml",
                        "price": "27.00",
                        "inStock": False,
                    },
                    {
                        "spinId": "spin-2",
                        "packSize": "1 L",
                        "price": "52.00",
                        "inStock": False,
                    },
                ],
            }
        )
    ]
    _patch_search_products(monkeypatch, products)

    result = await substitution.propose_substitute(
        _SESSION,
        _USER_ID,
        _ADDRESS_ID,
        _INGREDIENT,
        preferred_pack_size="500 ml",
        preferred_quantity=1,
    )

    assert result is None


async def test_all_priceless_returns_none(monkeypatch: Any) -> None:
    products = [
        Product.model_validate(
            {
                "productId": "prod-1",
                "name": "Amul Milk",
                "variants": [
                    {"spinId": "spin-1", "packSize": "500 ml", "inStock": True},
                    {"spinId": "spin-2", "packSize": "1 L", "inStock": True},
                ],
            }
        )
    ]
    _patch_search_products(monkeypatch, products)

    result = await substitution.propose_substitute(
        _SESSION,
        _USER_ID,
        _ADDRESS_ID,
        _INGREDIENT,
        preferred_pack_size="500 ml",
        preferred_quantity=1,
    )

    assert result is None


async def test_unparseable_pack_size_ranks_below_a_close_parseable_match(
    monkeypatch: Any,
) -> None:
    products = [
        Product.model_validate(
            {
                "productId": "prod-1",
                "name": "Sugar",
                "variants": [
                    {
                        "spinId": "spin-weird",
                        "packSize": "family pack",
                        "price": "20.00",
                    },
                    {"spinId": "spin-close", "packSize": "500 g", "price": "100.00"},
                ],
            }
        )
    ]
    _patch_search_products(monkeypatch, products)

    result = await substitution.propose_substitute(
        _SESSION,
        _USER_ID,
        _ADDRESS_ID,
        "sugar",
        preferred_pack_size="500 g",
        preferred_quantity=1,
    )

    # The parseable, exact pack-size match wins even though it's pricier.
    assert result is not None
    assert result.spin_id == "spin-close"


async def test_unparseable_pack_size_still_returned_when_only_purchasable_option(
    monkeypatch: Any,
) -> None:
    products = [
        Product.model_validate(
            {
                "productId": "prod-1",
                "name": "Sugar",
                "variants": [
                    {
                        "spinId": "spin-weird",
                        "packSize": "family pack",
                        "price": "20.00",
                    },
                ],
            }
        )
    ]
    _patch_search_products(monkeypatch, products)

    result = await substitution.propose_substitute(
        _SESSION,
        _USER_ID,
        _ADDRESS_ID,
        "sugar",
        preferred_pack_size="500 g",
        preferred_quantity=1,
    )

    assert result is not None
    assert result.spin_id == "spin-weird"


async def test_missing_pack_size_on_candidate_treated_as_unknown_distance(
    monkeypatch: Any,
) -> None:
    products = [
        Product.model_validate(
            {
                "productId": "prod-1",
                "name": "Sugar",
                "variants": [
                    {"spinId": "spin-no-size", "price": "20.00"},
                ],
            }
        )
    ]
    _patch_search_products(monkeypatch, products)

    result = await substitution.propose_substitute(
        _SESSION,
        _USER_ID,
        _ADDRESS_ID,
        "sugar",
        preferred_pack_size="500 g",
        preferred_quantity=1,
    )

    assert result is not None
    assert result.spin_id == "spin-no-size"
    assert result.pack_size is None


async def test_product_name_falls_back_to_brand_then_ingredient_name(
    monkeypatch: Any,
) -> None:
    no_name_no_brand = Product.model_validate(
        {
            "productId": "prod-1",
            "variants": [
                {"spinId": "spin-1", "packSize": "500 g", "price": "20.00"},
            ],
        }
    )
    _patch_search_products(monkeypatch, [no_name_no_brand])

    result = await substitution.propose_substitute(
        _SESSION,
        _USER_ID,
        _ADDRESS_ID,
        "sugar",
        preferred_pack_size="500 g",
        preferred_quantity=1,
    )

    assert result is not None
    assert result.product_name == "sugar"
    assert "sugar" in result.reason

    no_name_with_brand = Product.model_validate(
        {
            "productId": "prod-1",
            "brand": "Madhur",
            "variants": [
                {"spinId": "spin-1", "packSize": "500 g", "price": "20.00"},
            ],
        }
    )
    _patch_search_products(monkeypatch, [no_name_with_brand])

    result = await substitution.propose_substitute(
        _SESSION,
        _USER_ID,
        _ADDRESS_ID,
        "sugar",
        preferred_pack_size="500 g",
        preferred_quantity=1,
    )

    assert result is not None
    assert result.product_name == "Madhur"


def test_select_preferred_candidate_defers_to_first_result_with_no_brand_match() -> (
    None
):
    products = [
        Product.model_validate(
            {
                "productId": "p-generic",
                "name": "Generic Sugar",
                "brand": "Generic",
                "variants": [
                    {"spinId": "spin-generic", "packSize": "1 kg", "price": "49.00"}
                ],
            }
        ),
        Product.model_validate(
            {
                "productId": "p-madhur",
                "name": "Madhur Pure Sugar",
                "brand": "Madhur",
                "variants": [
                    {"spinId": "spin-madhur", "packSize": "1 kg", "price": "62.00"}
                ],
            }
        ),
    ]

    candidate = substitution.select_preferred_candidate(products)

    assert candidate is not None
    _, variant, _ = candidate
    assert variant.spin_id == "spin-generic"


def test_select_preferred_candidate_prefers_a_go_to_brand_over_first_result() -> None:
    products = [
        Product.model_validate(
            {
                "productId": "p-generic",
                "name": "Generic Sugar",
                "brand": "Generic",
                "variants": [
                    {"spinId": "spin-generic", "packSize": "1 kg", "price": "49.00"}
                ],
            }
        ),
        Product.model_validate(
            {
                "productId": "p-madhur",
                "name": "Madhur Pure Sugar",
                "brand": "Madhur",
                "variants": [
                    {"spinId": "spin-madhur", "packSize": "1 kg", "price": "62.00"}
                ],
            }
        ),
    ]

    candidate = substitution.select_preferred_candidate(
        products, preferred_brands=frozenset({"Madhur"})
    )

    assert candidate is not None
    _, variant, _ = candidate
    assert variant.spin_id == "spin-madhur"


def test_select_preferred_candidate_ignores_unmatched_preferred_brands() -> None:
    products = [
        Product.model_validate(
            {
                "productId": "p-generic",
                "name": "Generic Sugar",
                "brand": "Generic",
                "variants": [
                    {"spinId": "spin-generic", "packSize": "1 kg", "price": "49.00"}
                ],
            }
        )
    ]

    candidate = substitution.select_preferred_candidate(
        products, preferred_brands=frozenset({"Madhur"})
    )

    assert candidate is not None
    _, variant, _ = candidate
    assert variant.spin_id == "spin-generic"


def test_select_preferred_candidate_skips_out_of_stock_and_unpriced() -> None:
    products = [
        Product.model_validate(
            {
                "productId": "p-oos",
                "name": "Out Of Stock",
                "variants": [
                    {"spinId": "spin-oos", "packSize": "1 kg", "inStock": False}
                ],
            }
        ),
        Product.model_validate(
            {
                "productId": "p-unpriced",
                "name": "Unpriced",
                "variants": [{"spinId": "spin-unpriced", "packSize": "1 kg"}],
            }
        ),
        Product.model_validate(
            {
                "productId": "p-good",
                "name": "Good Sugar",
                "variants": [
                    {"spinId": "spin-good", "packSize": "1 kg", "price": "50.00"}
                ],
            }
        ),
    ]

    candidate = substitution.select_preferred_candidate(products)

    assert candidate is not None
    _, variant, _ = candidate
    assert variant.spin_id == "spin-good"


def test_select_preferred_candidate_returns_none_when_nothing_purchasable() -> None:
    products = [
        Product.model_validate(
            {
                "productId": "p-oos",
                "name": "Out Of Stock",
                "variants": [
                    {"spinId": "spin-oos", "packSize": "1 kg", "inStock": False}
                ],
            }
        )
    ]

    assert substitution.select_preferred_candidate(products) is None


async def test_deterministic_across_repeated_calls(monkeypatch: Any) -> None:
    products = [
        Product.model_validate(
            {
                "productId": "prod-1",
                "name": "Sugar",
                "variants": [
                    {"spinId": "spin-b", "packSize": "500 g", "price": "30.00"},
                    {"spinId": "spin-a", "packSize": "500 g", "price": "30.00"},
                    {"spinId": "spin-c", "packSize": "1 kg", "price": "10.00"},
                ],
            }
        )
    ]
    _patch_search_products(monkeypatch, products)

    first = await substitution.propose_substitute(
        _SESSION,
        _USER_ID,
        _ADDRESS_ID,
        "sugar",
        preferred_pack_size="500 g",
        preferred_quantity=1,
    )
    second = await substitution.propose_substitute(
        _SESSION,
        _USER_ID,
        _ADDRESS_ID,
        "sugar",
        preferred_pack_size="500 g",
        preferred_quantity=1,
    )

    assert first == second
    assert isinstance(first, SubstitutionResult)
