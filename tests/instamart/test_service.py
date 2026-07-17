from __future__ import annotations

from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.instamart import service
from app.instamart.exceptions import InstamartAuthError
from app.instamart.schemas import (
    AddressCategory,
    CreateAddressRequest,
    GoToItem,
    PreferenceSignal,
)
from app.models.user import User
from app.providers import service as provider_service
from tests.conftest import InstamartToolCallStub

RAW_ADDRESS = {
    "addressId": "addr-1",
    "fullAddress": "221B Baker Street, London",
    "addressLine": "221B Baker Street",
    "city": "London",
    "postalCode": "NW16XE",
    "addressCategory": "HOME",
}


async def test_get_addresses_parses_a_nested_list(
    db_session: AsyncSession,
    linked_user: User,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure(
        result={"success": True, "data": {"addresses": [RAW_ADDRESS]}}
    )

    addresses = await service.get_addresses(db_session, linked_user.id)

    assert len(addresses) == 1
    assert addresses[0].address_id == "addr-1"
    assert addresses[0].address_category == AddressCategory.HOME


async def test_get_addresses_accepts_a_bare_list_payload(
    db_session: AsyncSession,
    linked_user: User,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure(result={"success": True, "data": [RAW_ADDRESS]})

    addresses = await service.get_addresses(db_session, linked_user.id)

    assert len(addresses) == 1
    assert addresses[0].address_id == "addr-1"


async def test_get_addresses_raises_auth_error_when_not_linked(
    db_session: AsyncSession,
    user: User,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    with pytest.raises(InstamartAuthError):
        await service.get_addresses(db_session, user.id)

    assert mock_instamart_tool_call.calls == []


async def test_get_addresses_marks_link_expired_on_live_auth_failure(
    db_session: AsyncSession,
    linked_user: User,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure(status_code=401)

    with pytest.raises(InstamartAuthError):
        await service.get_addresses(db_session, linked_user.id)

    link = await provider_service.get_link(db_session, linked_user.id)
    assert link is not None
    assert link.status == "expired"


async def test_create_address_sends_documented_camelcase_arguments(
    db_session: AsyncSession,
    linked_user: User,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure(result={"success": True, "data": RAW_ADDRESS})
    request = CreateAddressRequest(
        full_address="221B Baker Street, London",
        address_line="221B Baker Street",
        city="London",
        postal_code="NW16XE",
        latitude=51.5237,
        longitude=-0.1585,
        address_category=AddressCategory.HOME,
        user_name="Sherlock Holmes",
        user_phone="+441234567890",
    )

    address = await service.create_address(db_session, linked_user.id, request)

    (_, kwargs) = mock_instamart_tool_call.calls[0]
    arguments = kwargs["json"]["params"]["arguments"]
    assert arguments["fullAddress"] == "221B Baker Street, London"
    assert arguments["addressCategory"] == "HOME"
    assert "address_category" not in arguments
    assert address.address_id == "addr-1"


async def test_delete_address_calls_the_tool_with_address_id(
    db_session: AsyncSession,
    linked_user: User,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure(result={"success": True, "data": {}})

    await service.delete_address(db_session, linked_user.id, "addr-1")

    (_, kwargs) = mock_instamart_tool_call.calls[0]
    assert kwargs["json"]["params"] == {
        "name": "delete_address",
        "arguments": {"addressId": "addr-1"},
    }


RAW_PRODUCT = {
    "productId": "prod-1",
    "name": "Amul Milk",
    "brand": "Amul",
    "variants": [
        {"spinId": "spin-1", "packSize": "500 ml", "price": "27.00", "inStock": True},
        {"spinId": "spin-2", "packSize": "1 L", "price": "52.00", "inStock": False},
    ],
}


async def test_search_products_scopes_the_call_to_the_selected_address(
    db_session: AsyncSession,
    linked_user: User,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure(
        result={"success": True, "data": {"products": []}}
    )

    await service.search_products(
        db_session, linked_user.id, address_id="addr-1", query="milk"
    )

    (_, kwargs) = mock_instamart_tool_call.calls[0]
    assert kwargs["json"]["params"] == {
        "name": "search_products",
        "arguments": {"addressId": "addr-1", "query": "milk", "offset": 0},
    }


async def test_search_products_parses_candidates_with_variant_level_fields(
    db_session: AsyncSession,
    linked_user: User,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure(
        result={"success": True, "data": {"products": [RAW_PRODUCT]}}
    )

    products = await service.search_products(
        db_session, linked_user.id, address_id="addr-1", query="milk"
    )

    assert len(products) == 1
    variants = products[0].variants
    assert [v.spin_id for v in variants] == ["spin-1", "spin-2"]
    assert variants[0].pack_size == "500 ml"
    assert variants[0].price == Decimal("27.00")
    assert variants[1].in_stock is False


async def test_search_products_accepts_a_bare_list_payload(
    db_session: AsyncSession,
    linked_user: User,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure(result={"success": True, "data": [RAW_PRODUCT]})

    products = await service.search_products(
        db_session, linked_user.id, address_id="addr-1", query="milk"
    )

    assert len(products) == 1


async def test_search_products_returns_an_empty_list_for_no_results(
    db_session: AsyncSession,
    linked_user: User,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure(
        result={"success": True, "data": {"products": []}}
    )

    products = await service.search_products(
        db_session, linked_user.id, address_id="addr-1", query="an-unmatched-query"
    )

    assert products == []


async def test_search_products_raises_auth_error_when_not_linked(
    db_session: AsyncSession,
    user: User,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    with pytest.raises(InstamartAuthError):
        await service.search_products(
            db_session, user.id, address_id="addr-1", query="milk"
        )

    assert mock_instamart_tool_call.calls == []


async def test_search_products_marks_link_expired_on_live_auth_failure(
    db_session: AsyncSession,
    linked_user: User,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure(status_code=419)

    with pytest.raises(InstamartAuthError):
        await service.search_products(
            db_session, linked_user.id, address_id="addr-1", query="milk"
        )

    link = await provider_service.get_link(db_session, linked_user.id)
    assert link is not None
    assert link.status == "expired"


RAW_GO_TO_ITEM = {
    "productName": "Amul Milk",
    "category": "Dairy",
    "variants": [
        {"spinId": "spin-1", "packSize": "500 ml", "price": "27.00"},
        {"spinId": "spin-2", "packSize": "1 L", "price": "52.00"},
    ],
}


async def test_get_go_to_items_sends_correct_params(
    db_session: AsyncSession,
    linked_user: User,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure(result={"success": True, "data": {"items": []}})

    await service.get_go_to_items(db_session, linked_user.id, "addr-1")

    (_, kwargs) = mock_instamart_tool_call.calls[0]
    assert kwargs["json"]["params"] == {
        "name": "your_go_to_items",
        "arguments": {"addressId": "addr-1", "offset": 0},
    }


async def test_get_go_to_items_parses_a_nested_list(
    db_session: AsyncSession,
    linked_user: User,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure(
        result={"success": True, "data": {"items": [RAW_GO_TO_ITEM]}}
    )

    items = await service.get_go_to_items(db_session, linked_user.id, "addr-1")

    assert len(items) == 1
    assert items[0].product_name == "Amul Milk"
    assert items[0].variants[0].spin_id == "spin-1"


async def test_get_go_to_items_accepts_a_bare_list_payload(
    db_session: AsyncSession,
    linked_user: User,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure(
        result={"success": True, "data": [RAW_GO_TO_ITEM]}
    )

    items = await service.get_go_to_items(db_session, linked_user.id, "addr-1")

    assert len(items) == 1
    assert items[0].product_name == "Amul Milk"


async def test_get_go_to_items_raises_auth_error_when_not_linked(
    db_session: AsyncSession,
    user: User,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    with pytest.raises(InstamartAuthError):
        await service.get_go_to_items(db_session, user.id, "addr-1")

    assert mock_instamart_tool_call.calls == []


async def test_get_go_to_items_marks_link_expired_on_live_auth_failure(
    db_session: AsyncSession,
    linked_user: User,
    mock_instamart_tool_call: InstamartToolCallStub,
) -> None:
    mock_instamart_tool_call.configure(status_code=401)

    with pytest.raises(InstamartAuthError):
        await service.get_go_to_items(db_session, linked_user.id, "addr-1")

    link = await provider_service.get_link(db_session, linked_user.id)
    assert link is not None
    assert link.status == "expired"


def test_normalize_preferences_maps_name_to_most_ordered_spin_id() -> None:
    item = GoToItem.model_validate(RAW_GO_TO_ITEM)

    preferences = service.normalize_preferences([item])

    assert preferences == {"Amul Milk": PreferenceSignal(spin_id="spin-1", brand=None)}


def test_normalize_preferences_skips_items_with_no_variants() -> None:
    item = GoToItem.model_validate({**RAW_GO_TO_ITEM, "variants": []})

    preferences = service.normalize_preferences([item])

    assert preferences == {}


def test_normalize_preferences_empty_list_returns_empty_dict() -> None:
    assert service.normalize_preferences([]) == {}


def test_normalize_preferences_keeps_first_occurrence_on_duplicate_names() -> None:
    first = GoToItem.model_validate(RAW_GO_TO_ITEM)
    duplicate = GoToItem.model_validate(
        {
            "productName": "Amul Milk",
            "variants": [{"spinId": "spin-9", "packSize": "200 ml"}],
        }
    )

    preferences = service.normalize_preferences([first, duplicate])

    assert preferences["Amul Milk"].spin_id == "spin-1"


def test_normalize_preferences_is_deterministic() -> None:
    items = [
        GoToItem.model_validate(RAW_GO_TO_ITEM),
        GoToItem.model_validate(
            {
                "productName": "Aashirvaad Atta",
                "variants": [{"spinId": "spin-atta-1"}],
            }
        ),
    ]

    first_result = service.normalize_preferences(items)
    second_result = service.normalize_preferences(items)

    assert first_result == second_result
