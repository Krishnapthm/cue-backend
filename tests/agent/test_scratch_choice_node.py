"""Tests for the verified ready-made versus scratch choice (CUE-96)."""

from __future__ import annotations

import uuid

import pytest
from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

from app.agent.context import CueContext
from app.agent.nodes import scratch_choice as node
from app.agent.schemas import (
    GeneratedRecipe,
    RecipeIngredient,
    RecipeStep,
    ScratchChoice,
    ScratchComponent,
)
from app.agent.state import AgentState
from app.instamart import service as instamart_service
from app.instamart.schemas import Product, ProductVariant


def _recipe() -> GeneratedRecipe:
    return GeneratedRecipe(
        dish_name="dosa",
        estimated_time_minutes=45,
        ingredients=[
            RecipeIngredient(name="rice", quantity=2, unit="cup"),
            RecipeIngredient(name="urad dal", quantity=0.5, unit="cup"),
            RecipeIngredient(name="fenugreek seeds", quantity=1, unit="tsp"),
            RecipeIngredient(name="potato", quantity=2),
        ],
        method_summary="Ferment the batter, then cook and fill the dosas.",
        steps=[
            RecipeStep(
                title="Cook it",
                instructions=["Combine everything and cook."],
            )
        ],
        scratch_components=[
            ScratchComponent(
                name="dosa batter",
                ready_made_name="dosa batter",
                constituent_names=["rice", "urad dal", "fenugreek seeds"],
            )
        ],
    )


def _state(recipe: GeneratedRecipe | None = None) -> AgentState:
    return {
        "session_id": "session-1",
        "user_id": 1,
        "messages": [HumanMessage(content="dosa")],
        "recipe": recipe or _recipe(),
    }


def _runtime() -> Runtime[CueContext]:
    return Runtime(
        context=CueContext(
            session=None,  # type: ignore[arg-type]
            user_id=1,
            chat_session_id=uuid.uuid4(),
            address_id="address-1",
        )
    )


def _product(*, in_stock: bool = True, name: str = "iD Fresh Dosa Batter") -> Product:
    return Product(
        name=name,
        variants=[ProductVariant(spin_id="spin-dosa", in_stock=in_stock)],
    )


async def test_finds_a_component_only_when_an_exact_ready_item_is_in_stock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _search(*_args: object, **_kwargs: object) -> list[Product]:
        return [_product()]

    monkeypatch.setattr(instamart_service, "search_products", _search)

    update = await node.find_scratch_component(_state(), _runtime())

    assert update["scratch_component"] is not None
    assert update["scratch_component"].ready_made_name == "dosa batter"


@pytest.mark.parametrize(
    ("products",),
    [([],), ([_product(in_stock=False)],), ([_product(name="idli batter")],)],
)
async def test_does_not_offer_an_unpurchasable_or_wrong_ready_item(
    monkeypatch: pytest.MonkeyPatch, products: list[Product]
) -> None:
    async def _search(*_args: object, **_kwargs: object) -> list[Product]:
        return products

    monkeypatch.setattr(instamart_service, "search_products", _search)

    assert (await node.find_scratch_component(_state(), _runtime())) == {
        "scratch_component": None
    }


async def test_does_not_offer_a_choice_about_something_the_user_already_has(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # "I already have dosa batter" - asking "ready-made, or from scratch?" is
    # the agent not listening, so the card never appears even though the
    # ready-made item is in stock.
    async def _search(*_args: object, **_kwargs: object) -> list[Product]:
        return [_product()]

    monkeypatch.setattr(instamart_service, "search_products", _search)

    recipe = _recipe()
    recipe = recipe.model_copy(
        update={
            "ingredients": [
                *recipe.ingredients,
                RecipeIngredient(name="dosa batter", user_supplied=True),
            ]
        }
    )

    assert (await node.find_scratch_component(_state(recipe), _runtime())) == {
        "scratch_component": None
    }


def test_scratch_choice_payload_has_a_discriminated_ui_variant() -> None:
    recipe = _recipe()
    state = _state(recipe)
    component = recipe.scratch_components[0]

    assert node.scratch_choice_payload(state, component) == {
        "ui": "scratch_choice",
        "dish_name": "dosa",
        "component_name": "dosa batter",
        "ready_made_name": "dosa batter",
        "options": [
            {"id": "ready_made", "label": "Use ready-made dosa batter"},
            {"id": "from_scratch", "label": "Make dosa batter from scratch"},
        ],
    }


def test_ready_made_replaces_all_component_constituents() -> None:
    recipe = _recipe()
    component = recipe.scratch_components[0]

    selected = node._recipe_for_choice(recipe, component, ScratchChoice.READY_MADE)

    assert [ingredient.name for ingredient in selected.ingredients] == [
        "potato",
        "dosa batter",
    ]


def test_from_scratch_keeps_constituents_and_omits_the_ready_item() -> None:
    recipe = _recipe()
    component = recipe.scratch_components[0]

    selected = node._recipe_for_choice(recipe, component, ScratchChoice.FROM_SCRATCH)

    assert selected == recipe


def test_a_recorded_choice_does_not_interrupt_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe = _recipe()
    state = _state(recipe)
    component = recipe.scratch_components[0]
    state["scratch_component"] = component
    state["scratch_choice"] = ScratchChoice.READY_MADE

    def _must_not_interrupt(_payload: object) -> object:
        raise AssertionError("a recorded choice must not be asked again")

    monkeypatch.setattr(node, "interrupt", _must_not_interrupt)

    update = node.choose_scratch_component(state)

    assert [ingredient.name for ingredient in update["recipe"].ingredients] == [
        "potato",
        "dosa batter",
    ]
