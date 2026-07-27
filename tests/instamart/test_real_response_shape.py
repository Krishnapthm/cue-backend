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
    # `offerPrice` is what the user actually pays, so it wins over `mrp`.
    assert second.variants[0].price == Decimal("240")
    assert second.variants[1].price == Decimal("123")


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
