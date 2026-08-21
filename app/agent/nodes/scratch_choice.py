"""Choose one verified ready-made component before listing ingredients."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime
from langgraph.types import interrupt

from app.agent.context import CueContext
from app.agent.nodes.recipe import render_recipe
from app.agent.schemas import (
    GeneratedRecipe,
    RecipeIngredient,
    ScratchChoice,
    ScratchChoiceDecision,
    ScratchChoiceInterrupt,
    ScratchChoiceOption,
    ScratchComponent,
)
from app.agent.state import AgentState
from app.instamart import service as instamart_service
from app.instamart.schemas import Product
from app.pantry.service import normalize_name

logger = logging.getLogger(__name__)


def _is_purchasable(component: ScratchComponent, products: list[Product]) -> bool:
    """Return whether search found an in-stock product for this exact component."""
    ready_name = normalize_name(component.ready_made_name)
    return any(
        product.name is not None
        and ready_name in normalize_name(product.name)
        and any(variant.in_stock for variant in product.variants)
        for product in products
    )


def _candidate_components(recipe: GeneratedRecipe) -> list[ScratchComponent]:
    """Return valid decomposition points, most significant first.

    A component the user already has is not a decomposition point: asking
    "ready-made idli batter, or make it from scratch?" of someone who opened
    with "I already have idli batter" is the agent not listening. The prompt
    asks the model not to emit one in that case; this is the floor under it,
    and it reads the same `user_supplied` flag the checklist does.
    """
    ingredient_names = {
        normalize_name(ingredient.name) for ingredient in recipe.ingredients
    }
    supplied = {
        normalize_name(ingredient.name)
        for ingredient in recipe.ingredients
        if ingredient.user_supplied
    }
    candidates = [
        component
        for component in recipe.scratch_components
        if len(set(map(normalize_name, component.constituent_names))) >= 2
        and set(map(normalize_name, component.constituent_names)) <= ingredient_names
        and normalize_name(component.ready_made_name) not in supplied
        and normalize_name(component.name) not in supplied
        and not (set(map(normalize_name, component.constituent_names)) & supplied)
    ]
    return sorted(
        candidates,
        key=lambda component: len(component.constituent_names),
        reverse=True,
    )


async def find_scratch_component(
    state: AgentState, runtime: Runtime[CueContext]
) -> dict[str, ScratchComponent | None]:
    """Find one address-available ready-made component, if a real choice exists."""
    recipe = state.get("recipe")
    if recipe is None:
        raise ValueError("Cannot find scratch choice: state has no recipe.")

    for component in _candidate_components(recipe):
        products = await instamart_service.search_products(
            runtime.context.session,
            runtime.context.user_id,
            address_id=runtime.context.address_id,
            query=component.ready_made_name,
        )
        if _is_purchasable(component, products):
            return {"scratch_component": component}

    logger.info(
        "No purchasable ready-made component for session %s.", state["session_id"]
    )
    return {"scratch_component": None}


def scratch_choice_payload(
    state: AgentState, component: ScratchComponent
) -> dict[str, Any]:
    """Build the JSON-serializable scratch-choice interrupt payload."""
    recipe = state.get("recipe")
    if recipe is None:
        raise ValueError("Cannot build scratch choice: state has no recipe.")
    return ScratchChoiceInterrupt(
        dish_name=recipe.dish_name,
        component_name=component.name,
        ready_made_name=component.ready_made_name,
        options=[
            ScratchChoiceOption(
                id=ScratchChoice.READY_MADE,
                label=f"Use ready-made {component.ready_made_name}",
            ),
            ScratchChoiceOption(
                id=ScratchChoice.FROM_SCRATCH,
                label=f"Make {component.name} from scratch",
            ),
        ],
    ).model_dump(mode="json")


def _recipe_for_choice(
    recipe: GeneratedRecipe, component: ScratchComponent, choice: ScratchChoice
) -> GeneratedRecipe:
    """Replace component constituents with the ready-made item when selected."""
    if choice is ScratchChoice.FROM_SCRATCH:
        return recipe
    constituent_names = {normalize_name(name) for name in component.constituent_names}
    ready_name = normalize_name(component.ready_made_name)
    if any(
        normalize_name(ingredient.name) == ready_name
        for ingredient in recipe.ingredients
    ):
        return recipe
    ingredients = [
        ingredient
        for ingredient in recipe.ingredients
        if normalize_name(ingredient.name) not in constituent_names
    ]
    ingredients.append(RecipeIngredient(name=component.ready_made_name))
    return recipe.model_copy(
        update={"ingredients": ingredients, "scratch_components": []}
    )


def choose_scratch_component(state: AgentState) -> dict[str, Any]:
    """Pause for the source choice, then render the selected ingredient list."""
    component = state.get("scratch_component")
    recipe = state.get("recipe")
    if recipe is None:
        raise ValueError("Cannot choose scratch component: state has no recipe.")
    if component is None:
        return {"messages": [AIMessage(content=render_recipe(recipe))]}
    if (choice := state.get("scratch_choice")) is not None:
        return {"recipe": _recipe_for_choice(recipe, component, choice)}

    decision = ScratchChoiceDecision.model_validate(
        interrupt(scratch_choice_payload(state, component))
    )
    selected_recipe = _recipe_for_choice(recipe, component, decision.choice)
    return {
        "recipe": selected_recipe,
        "scratch_choice": decision.choice,
        "messages": [AIMessage(content=render_recipe(selected_recipe))],
    }
