"""Coverage against Swiggy's *real* wire format (CUE-77).

Every other Instamart test mocks the response as
`{"result": {"structuredContent": ...}}` - a shape the client assumed but
Swiggy never actually sends. That fiction is why `call_tool` returned None on
every production call while 400+ tests stayed green. The payloads here are
trimmed copies of responses logged against a live linked account, so they keep
Swiggy's real field names: the envelope is JSON *text* inside a content block,
products come back under `variations` (not `variants`) with
`quantityDescription` and a nested `price` object.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.instamart import client, service
from app.instamart.exceptions import InstamartDomainError
from app.instamart.schemas import Cart
from app.models.user import User
from tests.conftest import InstamartToolCallStub

# A trimmed copy of a real `search_products` response for "sugar": two
# products, the second with two variations, so variant selection has a real
# choice to make.
REAL_SEARCH_DATA = {
    "nextOffset": "1",
    "products": [
        {
            "displayName": "Supreme Harvest Crystal Sugar",
            "brand": "Supreme Harvest",
            "inStock": True,
            "isAvail": True,
            "productId": "2UDNQI8MR0",
            "parentProductId": "SX1UPX5JTP",
            "isPromoted": False,
            "variations": [
                {
                    "spinId": "98DG1XN9O2",
                    "skuId": "MVXSBNQV9A",
                    "quantityDescription": "1 kg",
                    "displayName": "Supreme Harvest Crystal Sugar",
                    "brandName": "Supreme Harvest",
                    "price": {
                        "mrp": 80,
                        "offerPrice": 80,
                        "unitLevelPrice": "8/100 g",
                    },
                    "isInStockAndAvailable": True,
                    "imageUrl": "https://media-assets.swiggy.com/swiggy/image/upload/sugar.png",
                    "rating": {"value": "4.5", "count": "51.5k"},
                    "vegClassifier": "VEG_CLASSIFIER_VEG",
                }
            ],
        },
        {
            "displayName": "24 Mantra Organic Sugar",
            "brand": "24 Mantra",
            "inStock": True,
            "isAvail": True,
            "productId": "MLAGDZDD8Z",
            "isPromoted": True,
            "variations": [
                {
                    "spinId": "3RTD0J61B4",
                    "quantityDescription": "1 kg x 2",
                    "price": {"mrp": 320, "offerPrice": 240},
                    "isInStockAndAvailable": True,
                    "imageUrl": "https://media-assets.swiggy.com/swiggy/image/upload/two-kg.png",
                },
                {
                    "spinId": "P7NDAXSABN",
                    "quantityDescription": "1 kg",
                    "price": {"mrp": 160, "offerPrice": 123},
                    "isInStockAndAvailable": True,
                },
            ],
        },
    ],
}

# A trimmed copy of a real `your_go_to_items` response.
REAL_GO_TO_DATA = {
    "items": [
        {
            "displayName": "Aashirvaad Shudh Chakki Atta",
            "brand": "Aashirvaad",
            "variations": [
                {
                    "spinId": "GOTO1SPIN",
                    "quantityDescription": "5 kg",
                    "price": {"mrp": 320, "offerPrice": 285},
                    "isInStockAndAvailable": True,
                }
            ],
        }
    ]
}


async def test_call_tool_unwraps_the_real_json_in_text_envelope(
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    """The bug: no `structuredContent`, no `isError`, payload as JSON text."""
    mock_instamart_tool_call.configure_text_envelope(
        {"success": True, "data": REAL_SEARCH_DATA, "message": "Found 20 products"}
    )

    data = await client.call_tool("at_token", "search_products", {})

    assert data == REAL_SEARCH_DATA


async def test_call_tool_raises_domain_error_on_success_false(
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    """`success: false` is the failure signal that actually fires, not `isError`."""
    mock_instamart_tool_call.configure_text_envelope(
        {
            "success": False,
            "error": {"message": "Item is out of stock in your area"},
        }
    )

    with pytest.raises(InstamartDomainError) as exc_info:
        await client.call_tool("at_token", "search_products", {})

    assert "out of stock in your area" in str(exc_info.value)


async def test_call_tool_falls_back_to_a_top_level_failure_message(
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    """A failure without the documented `error` object still surfaces text."""
    mock_instamart_tool_call.configure_text_envelope(
        {"success": False, "message": "Beta users only"}
    )

    with pytest.raises(InstamartDomainError) as exc_info:
        await client.call_tool("at_token", "search_products", {})

    assert "Beta users only" in str(exc_info.value)


async def test_call_tool_still_prefers_structured_content_when_present(
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    """Forward compatibility: the old shape must keep working untouched."""
    mock_instamart_tool_call.configure(result={"structuredContent": {"addresses": []}})

    data = await client.call_tool("at_token", "get_addresses", {})

    assert data == {"addresses": []}


async def test_call_tool_returns_none_for_a_non_json_text_block(
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    """A prose-only result is not an error - it just carries no payload."""
    mock_instamart_tool_call.configure(
        result={"content": [{"type": "text", "text": "not json at all"}]}
    )

    assert await client.call_tool("at_token", "search_products", {}) is None


async def test_search_products_parses_the_real_shape_end_to_end(
    db_session: AsyncSession,
    linked_user: User,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    """The acceptance path: a real "sugar" search yields purchasable variants."""
    mock_instamart_tool_call.configure_text_envelope(
        {"success": True, "data": REAL_SEARCH_DATA}
    )

    products = await service.search_products(
        db_session, linked_user.id, address_id="addr-1", query="sugar"
    )

    assert [product.name for product in products] == [
        "Supreme Harvest Crystal Sugar",
        "24 Mantra Organic Sugar",
    ]
    assert [product.brand for product in products] == ["Supreme Harvest", "24 Mantra"]

    # The regression that made every tap resolve `unresolved`: variants parsed
    # empty, so nothing carried a spinId to put in a cart.
    first, second = products
    assert [variant.spin_id for variant in first.variants] == ["98DG1XN9O2"]
    assert [variant.spin_id for variant in second.variants] == [
        "3RTD0J61B4",
        "P7NDAXSABN",
    ]
    assert first.variants[0].pack_size == "1 kg"
    assert first.variants[0].in_stock is True
    assert first.variants[0].image_url == (
        "https://media-assets.swiggy.com/swiggy/image/upload/sugar.png"
    )
    assert first.variants[0].rating is not None
    assert first.variants[0].rating.value == "4.5"
    assert first.variants[0].rating.count == "51.5k"
    # `offerPrice` is what the user actually pays, so it wins over `mrp`.
    assert second.variants[0].price == Decimal("240")
    assert second.variants[1].price == Decimal("123")
    # The first variation has only an image, while the second has neither.
    # Both are normal optional cases in the observed search payload.
    assert second.variants[0].image_url == (
        "https://media-assets.swiggy.com/swiggy/image/upload/two-kg.png"
    )
    assert second.variants[0].rating is None
    assert second.variants[1].image_url is None
    assert second.variants[1].rating is None


def test_cart_line_item_resolves_price_from_flat_siblings() -> None:
    """A real linked-account `get_cart` capture has no `price` field at all -
    a line prices itself with flat `mrp`/`discountedFinalPrice` siblings
    instead, unlike `search_products`'s nested `{mrp, offerPrice}` object. A
    cart line that doesn't resolve one from the other parses with
    `price=None`, which reads to the user as Rs 0 for every chat-composed
    item - `discountedFinalPrice` is what Swiggy actually charges."""
    line = {
        "spinId": "spin-1",
        "quantity": 2,
        "mrp": 90,
        "discountedFinalPrice": 69,
    }

    cart = Cart.model_validate({"items": [line]})

    assert cart.items[0].price == Decimal("69")


def test_cart_line_item_falls_back_to_mrp_with_no_discount() -> None:
    line = {"spinId": "spin-1", "quantity": 1, "mrp": 34, "discountedFinalPrice": 34}

    cart = Cart.model_validate({"items": [line]})

    assert cart.items[0].price == Decimal("34")


def test_cart_line_item_still_accepts_an_explicit_price_field() -> None:
    """Not just a real capture's shape - an explicit `price` scalar, if
    Swiggy ever sends one, is trusted over the flat siblings rather than
    ignored."""
    cart = Cart.model_validate(
        {"items": [{"spinId": "spin-1", "quantity": 1, "price": "27.00"}]}
    )

    assert cart.items[0].price == Decimal("27.00")


def test_cart_parses_a_real_get_cart_response() -> None:
    """A trimmed copy of a real `get_cart` response logged against a live
    linked account: flat per-line `mrp`/`discountedFinalPrice`, no nested
    `price` object anywhere, and a display-string cart total keyed
    `cartTotalAmount` rather than `total`."""
    raw_cart = {
        "cartTotalAmount": "₹760",
        "items": [
            {
                "spinId": "SHZR8VDLRJ",
                "itemName": "Daawat Pulav Basmati Rice",
                "quantity": 1,
                "mrp": 90,
                "discountedFinalPrice": 69,
            },
            {
                "spinId": "OPX8FP6RWK",
                "itemName": "Coconut Chunks (Thengai Thundugal)",
                "quantity": 2,
                "mrp": 83,
                "discountedFinalPrice": 63,
            },
        ],
    }

    cart = Cart.model_validate(raw_cart)

    assert cart.total == Decimal("760")
    assert cart.items[0].price == Decimal("69")
    assert cart.items[0].product_name == "Daawat Pulav Basmati Rice"
    assert cart.items[1].price == Decimal("63")


def test_cart_line_items_preserve_optional_variant_metadata() -> None:
    """Cart endpoint models expose metadata when Swiggy includes it."""
    both = {
        "spinId": "both",
        "quantity": 1,
        "imageUrl": "https://media-assets.swiggy.com/swiggy/image/upload/both.png",
        "rating": {"value": "4.6", "count": "9.8k"},
    }
    only_image = {
        "spinId": "only-image",
        "quantity": 1,
        "imageUrl": "https://media-assets.swiggy.com/swiggy/image/upload/image.png",
    }
    neither = {"spinId": "neither", "quantity": 1}

    cart = Cart.model_validate({"items": [both, only_image, neither]})

    assert cart.items[0].rating is not None
    assert cart.items[0].rating.value == "4.6"
    assert cart.items[0].rating.count == "9.8k"
    assert cart.items[1].image_url == only_image["imageUrl"]
    assert cart.items[1].rating is None
    assert cart.items[2].image_url is None
    assert cart.items[2].rating is None


async def test_get_go_to_items_parses_the_real_shape_end_to_end(
    db_session: AsyncSession,
    linked_user: User,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure_text_envelope(
        {"success": True, "data": REAL_GO_TO_DATA}
    )

    items = await service.get_go_to_items(db_session, linked_user.id, "addr-1")

    assert len(items) == 1
    assert items[0].variants[0].spin_id == "GOTO1SPIN"


async def test_search_products_surfaces_a_domain_error_from_the_real_shape(
    db_session: AsyncSession,
    linked_user: User,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure_text_envelope(
        {"success": False, "error": {"message": "Address not serviceable"}}
    )

    with pytest.raises(InstamartDomainError):
        await service.search_products(
            db_session, linked_user.id, address_id="addr-1", query="sugar"
        )
