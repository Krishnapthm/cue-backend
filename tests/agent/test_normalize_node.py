"""`normalize_ingredients_node`: have/need marking and duplicate-name merge
rules.

These are pure unit tests - no model, no network. `AgentState` literals are
built directly with a `GeneratedRecipe`, mirroring `tests/agent/
test_recipe_node.py`'s style.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from app.agent.nodes.normalize import normalize_ingredients_node
from app.agent.schemas import (
    GeneratedRecipe,
    IngredientStatus,
    NormalizedIngredient,
    RecipeIngredient,
)
from app.agent.state import AgentState


def _recipe(ingredients: list[RecipeIngredient]) -> GeneratedRecipe:
    return GeneratedRecipe(
        dish_name="test dish",
        estimated_time_minutes=10,
        ingredients=ingredients,
        method_summary="Combine everything.",
    )


def _state(
    ingredients: list[RecipeIngredient],
    have_marks: set[str] | None = None,
) -> AgentState:
    state: AgentState = {
        "session_id": "session-1",
        "user_id": 1,
        "messages": [HumanMessage(content="test dish")],
        "recipe": _recipe(ingredients),
    }
    if have_marks is not None:
        state["have_marks"] = have_marks
    return state


def test_basic_have_and_need_split() -> None:
    ingredients = [
        RecipeIngredient(name="salt", quantity=1, unit="tsp"),
        RecipeIngredient(name="pepper", quantity=1, unit="tsp"),
        RecipeIngredient(name="sugar", quantity=2, unit="tbsp"),
    ]
    state = _state(ingredients, have_marks={"pepper"})

    update = normalize_ingredients_node(state)

    normalized = update["normalized_ingredients"]
    assert len(normalized) == 3
    by_name = {row.name: row for row in normalized}
    assert by_name["pepper"].status == IngredientStatus.HAVE
    assert by_name["salt"].status == IngredientStatus.NEED
    assert by_name["sugar"].status == IngredientStatus.NEED

    need_only = [row for row in normalized if row.status == IngredientStatus.NEED]
    assert len(need_only) == 2
    assert {row.name for row in need_only} == {"salt", "sugar"}


def test_have_marks_entry_matching_nothing_is_ignored() -> None:
    ingredients = [
        RecipeIngredient(name="salt", quantity=1, unit="tsp"),
        RecipeIngredient(name="pepper", quantity=1, unit="tsp"),
    ]
    state = _state(ingredients, have_marks={"cardamom pods"})

    update = normalize_ingredients_node(state)

    normalized = update["normalized_ingredients"]
    assert len(normalized) == 2
    assert all(row.status == IngredientStatus.NEED for row in normalized)


def test_duplicate_name_same_unit_merges_and_sums_quantities() -> None:
    ingredients = [
        RecipeIngredient(name="salt", quantity=1, unit="tsp"),
        RecipeIngredient(name="pepper", quantity=1, unit="tsp"),
        RecipeIngredient(name="salt", quantity=2, unit="tsp"),
    ]
    state = _state(ingredients)

    update = normalize_ingredients_node(state)

    normalized = update["normalized_ingredients"]
    salt_rows = [row for row in normalized if row.name == "salt"]
    assert len(salt_rows) == 1
    assert salt_rows[0].quantity == 3
    assert salt_rows[0].unit == "tsp"


def test_duplicate_name_different_units_kept_separate() -> None:
    ingredients = [
        RecipeIngredient(name="salt", quantity=1, unit="tsp"),
        RecipeIngredient(name="salt", quantity=1, unit="g"),
    ]
    state = _state(ingredients)

    update = normalize_ingredients_node(state)

    normalized = update["normalized_ingredients"]
    salt_rows = [row for row in normalized if row.name == "salt"]
    assert len(salt_rows) == 2
    assert {row.unit for row in salt_rows} == {"tsp", "g"}
    assert [row.quantity for row in salt_rows] == [1.0, 1.0]


def test_duplicate_name_both_units_none_kept_separate() -> None:
    ingredients = [
        RecipeIngredient(name="salt", quantity=1, unit=None),
        RecipeIngredient(name="salt", quantity=2, unit=None),
    ]
    state = _state(ingredients)

    update = normalize_ingredients_node(state)

    normalized = update["normalized_ingredients"]
    salt_rows = [row for row in normalized if row.name == "salt"]
    assert len(salt_rows) == 2
    assert all(row.unit is None for row in salt_rows)
    assert sorted(row.quantity for row in salt_rows) == [1.0, 2.0]


def test_duplicate_name_same_unit_one_quantity_none_merges_to_the_number() -> None:
    ingredients = [
        RecipeIngredient(name="salt", quantity=None, unit="tsp"),
        RecipeIngredient(name="salt", quantity=5, unit="tsp"),
    ]
    state = _state(ingredients)

    update = normalize_ingredients_node(state)

    normalized = update["normalized_ingredients"]
    salt_rows = [row for row in normalized if row.name == "salt"]
    assert len(salt_rows) == 1
    assert salt_rows[0].quantity == 5


def test_duplicate_name_same_unit_all_quantities_none_stays_none() -> None:
    ingredients = [
        RecipeIngredient(name="salt", quantity=None, unit="tsp"),
        RecipeIngredient(name="salt", quantity=None, unit="tsp"),
    ]
    state = _state(ingredients)

    update = normalize_ingredients_node(state)

    normalized = update["normalized_ingredients"]
    salt_rows = [row for row in normalized if row.name == "salt"]
    assert len(salt_rows) == 1
    assert salt_rows[0].quantity is None


def test_output_order_is_stable_first_appearance() -> None:
    ingredients = [
        RecipeIngredient(name="salt", quantity=1, unit="tsp"),
        RecipeIngredient(name="pepper", quantity=1, unit="tsp"),
        RecipeIngredient(name="salt", quantity=1, unit="g"),
        RecipeIngredient(name="sugar", quantity=1, unit="tbsp"),
        RecipeIngredient(name="salt", quantity=1, unit="tsp"),
    ]
    state = _state(ingredients)

    update = normalize_ingredients_node(state)
    normalized = update["normalized_ingredients"]

    # salt's name is first-seen before pepper and sugar, so all of salt's
    # rows (grouped, unit-ordered by first appearance: tsp then g) come
    # before pepper and sugar.
    assert [(row.name, row.unit) for row in normalized] == [
        ("salt", "tsp"),
        ("salt", "g"),
        ("pepper", "tsp"),
        ("sugar", "tbsp"),
    ]

    # Determinism: running again on the same inputs yields an identical
    # result.
    update_again = normalize_ingredients_node(state)
    assert update_again["normalized_ingredients"] == normalized


def test_empty_have_marks_all_need() -> None:
    ingredients = [RecipeIngredient(name="salt", quantity=1, unit="tsp")]
    state = _state(ingredients, have_marks=set())

    update = normalize_ingredients_node(state)

    assert update["normalized_ingredients"][0].status == IngredientStatus.NEED


def test_missing_have_marks_key_all_need() -> None:
    ingredients = [RecipeIngredient(name="salt", quantity=1, unit="tsp")]
    state = _state(ingredients)
    assert "have_marks" not in state

    update = normalize_ingredients_node(state)

    assert update["normalized_ingredients"][0].status == IngredientStatus.NEED


def test_raises_on_missing_recipe() -> None:
    state: AgentState = {
        "session_id": "session-1",
        "user_id": 1,
        "messages": [HumanMessage(content="test dish")],
    }

    with pytest.raises(ValueError, match="no recipe"):
        normalize_ingredients_node(state)


def test_returns_partial_update_shape() -> None:
    ingredients = [RecipeIngredient(name="salt", quantity=1, unit="tsp")]
    state = _state(ingredients)

    update = normalize_ingredients_node(state)

    assert set(update.keys()) == {"normalized_ingredients"}
    normalized = update["normalized_ingredients"]
    assert isinstance(normalized, list)
    assert all(isinstance(row, NormalizedIngredient) for row in normalized)
