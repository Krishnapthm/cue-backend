"""`normalize_ingredients_node`: pantry seeding, and the duplicate-name merge
rules.

The merge rules are pure logic - no model, no network. The pantry seed is not:
it reads the user's real `pantry_item` rows, so these run against the suite's
ephemeral Postgres rather than a stubbed service. Every test therefore starts
from a `user` whose pantry is empty and adds only what it needs, which is also
what keeps them independent of each other on a database that is never
truncated between tests.
"""

from __future__ import annotations

import uuid
from typing import Any, cast

import pytest
from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.context import CueContext
from app.agent.nodes.confirm_checklist import checklist_payload
from app.agent.nodes.match_ingredient import needed_ingredients
from app.agent.nodes.normalize import ASSUMED_STAPLES, normalize_ingredients_node
from app.agent.schemas import (
    GeneratedRecipe,
    IngredientStatus,
    NormalizedIngredient,
    RecipeIngredient,
)
from app.agent.state import AgentState
from app.models.pantry import PantryItem
from app.models.user import User
from app.pantry.constants import LEVEL_MAX, LEVEL_MIN, PantryCategory
from app.pantry.service import normalize_name


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


def _runtime(session: AsyncSession, user_id: int) -> Runtime[CueContext]:
    """The runtime context an invocation supplies; the node reads the session."""
    return Runtime(
        context=CueContext(
            session=session,
            user_id=user_id,
            chat_session_id=uuid.uuid4(),
            address_id="addr-1",
        )
    )


async def _stock(
    session: AsyncSession, user_id: int, name: str, level: int = LEVEL_MAX
) -> None:
    """Put one staple in a user's pantry, at full level unless told otherwise."""
    session.add(
        PantryItem(
            user_id=user_id,
            name=name,
            name_normalized=normalize_name(name),
            category=PantryCategory.SPICES_AND_MASALAS.value,
            level=level,
        )
    )
    await session.commit()


@pytest.fixture
def flour_pepper_sugar() -> list[RecipeIngredient]:
    return [
        RecipeIngredient(name="flour", quantity=1, unit="cup"),
        RecipeIngredient(name="pepper", quantity=1, unit="tsp"),
        RecipeIngredient(name="sugar", quantity=2, unit="tbsp"),
    ]


# --- assumed staples -------------------------------------------------------


def test_assumed_staples_are_intentionally_narrow() -> None:
    assert frozenset({"salt", "water"}) == ASSUMED_STAPLES
    assert {"cooking oil", "sugar", "cumin", "black pepper"}.isdisjoint(
        ASSUMED_STAPLES
    )


async def test_assumed_staples_never_reach_the_checklist_or_cart_path() -> None:
    state = _state(
        [
            RecipeIngredient(name="salt", quantity=1, unit="tsp"),
            RecipeIngredient(name=" Water ", quantity=250, unit="ml"),
            RecipeIngredient(name="water", quantity=1, unit="kg"),
            RecipeIngredient(name="cooking oil", quantity=2, unit="tbsp"),
            RecipeIngredient(name="sugar", quantity=1, unit="tbsp"),
            RecipeIngredient(name="cumin", quantity=1, unit="tsp"),
        ],
        have_marks={"unrelated"},
    )

    runtime: Any = None
    update = await normalize_ingredients_node(state, runtime)
    normalized_state = cast(AgentState, {**state, **update})

    assert [row.name for row in update["normalized_ingredients"]] == [
        "water",
        "cooking oil",
        "sugar",
        "cumin",
    ]
    assert [item["name"] for item in checklist_payload(normalized_state)["items"]] == [
        "water",
        "cooking oil",
        "sugar",
        "cumin",
    ]
    assert [row.name for row in needed_ingredients(normalized_state)] == [
        "water",
        "cooking oil",
        "sugar",
        "cumin",
    ]


# --- the pantry seed -------------------------------------------------------


async def test_pantry_seeds_have_marks_with_no_client_input(
    db_session: AsyncSession, user: User, flour_pepper_sugar: list[RecipeIngredient]
) -> None:
    await _stock(db_session, user.id, "Pepper")
    state = _state(flour_pepper_sugar)
    assert "have_marks" not in state

    update = await normalize_ingredients_node(state, _runtime(db_session, user.id))

    by_name = {row.name: row for row in update["normalized_ingredients"]}
    assert by_name["pepper"].status == IngredientStatus.HAVE
    assert by_name["flour"].status == IngredientStatus.NEED
    assert by_name["sugar"].status == IngredientStatus.NEED
    # The seed is spelled as the *recipe* spells it, not as the pantry does -
    # every downstream reader matches on ingredient names.
    assert update["have_marks"] == {"pepper"}


async def test_the_seed_matches_case_and_whitespace_insensitively(
    db_session: AsyncSession, user: User
) -> None:
    await _stock(db_session, user.id, "  BASMATI   Rice ")
    state = _state([RecipeIngredient(name="Basmati Rice", quantity=200, unit="g")])

    update = await normalize_ingredients_node(state, _runtime(db_session, user.id))

    assert update["normalized_ingredients"][0].status == IngredientStatus.HAVE


async def test_a_near_miss_name_is_left_as_need(
    db_session: AsyncSession, user: User
) -> None:
    # "rice" is a substring of the staple, and a looser matcher would tick it.
    # Under-matching is the correct direction to be wrong in: a missed match
    # costs one tick, a false HAVE silently drops an ingredient from the order.
    await _stock(db_session, user.id, "basmati rice")
    state = _state([RecipeIngredient(name="rice", quantity=200, unit="g")])

    update = await normalize_ingredients_node(state, _runtime(db_session, user.id))

    assert update["normalized_ingredients"][0].status == IngredientStatus.NEED
    assert update["have_marks"] == set()


async def test_an_out_of_stock_staple_does_not_seed(
    db_session: AsyncSession, user: User
) -> None:
    # level 0 is "Out", which is precisely the staple that still needs buying.
    await _stock(db_session, user.id, "flour", level=LEVEL_MIN)
    state = _state([RecipeIngredient(name="flour", quantity=1, unit="cup")])

    update = await normalize_ingredients_node(state, _runtime(db_session, user.id))

    assert update["normalized_ingredients"][0].status == IngredientStatus.NEED


async def test_a_low_but_nonzero_staple_seeds(
    db_session: AsyncSession, user: User
) -> None:
    await _stock(db_session, user.id, "flour", level=LEVEL_MIN + 1)
    state = _state([RecipeIngredient(name="flour", quantity=1, unit="cup")])

    update = await normalize_ingredients_node(state, _runtime(db_session, user.id))

    assert update["normalized_ingredients"][0].status == IngredientStatus.HAVE


async def test_an_empty_pantry_leaves_every_row_need(
    db_session: AsyncSession, user: User, flour_pepper_sugar: list[RecipeIngredient]
) -> None:
    update = await normalize_ingredients_node(
        _state(flour_pepper_sugar), _runtime(db_session, user.id)
    )

    normalized = update["normalized_ingredients"]
    assert len(normalized) == 3
    assert all(row.status == IngredientStatus.NEED for row in normalized)
    assert update["have_marks"] == set()


async def test_another_users_pantry_never_seeds(
    db_session: AsyncSession, user: User
) -> None:
    other = User(firebase_uid=f"firebase-uid-{uuid.uuid4()}", email="other@example.com")
    db_session.add(other)
    await db_session.commit()
    await _stock(db_session, other.id, "flour")

    update = await normalize_ingredients_node(
        _state([RecipeIngredient(name="flour", quantity=1, unit="cup")]),
        _runtime(db_session, user.id),
    )

    assert update["normalized_ingredients"][0].status == IngredientStatus.NEED


# --- the user's marks win --------------------------------------------------


async def test_user_supplied_marks_override_the_pantry_seed(
    db_session: AsyncSession, user: User, flour_pepper_sugar: list[RecipeIngredient]
) -> None:
    # The pantry says pepper is in stock; the user's answer says only sugar is.
    # Re-applying the seed on top would re-tick a staple the user unticked
    # because they had run out - and they would never get the pepper.
    await _stock(db_session, user.id, "pepper")
    state = _state(flour_pepper_sugar, have_marks={"sugar"})

    update = await normalize_ingredients_node(state, _runtime(db_session, user.id))

    by_name = {row.name: row for row in update["normalized_ingredients"]}
    assert by_name["sugar"].status == IngredientStatus.HAVE
    assert by_name["pepper"].status == IngredientStatus.NEED
    assert by_name["flour"].status == IngredientStatus.NEED
    assert update["have_marks"] == {"sugar"}


async def test_empty_marks_are_treated_as_no_answer_and_seed(
    db_session: AsyncSession, user: User
) -> None:
    # An empty set is the absence of an answer, not an answer of "none of them":
    # `normalize_ingredients_node` is only reached before the user has been
    # asked, so there is no answer to preserve here.
    await _stock(db_session, user.id, "flour")
    state = _state([RecipeIngredient(name="flour", quantity=1, unit="cup")], set())

    update = await normalize_ingredients_node(state, _runtime(db_session, user.id))

    assert update["normalized_ingredients"][0].status == IngredientStatus.HAVE


async def test_have_marks_entry_matching_nothing_is_ignored(
    db_session: AsyncSession, user: User
) -> None:
    ingredients = [
        RecipeIngredient(name="flour", quantity=1, unit="cup"),
        RecipeIngredient(name="pepper", quantity=1, unit="tsp"),
    ]
    state = _state(ingredients, have_marks={"cardamom pods"})

    update = await normalize_ingredients_node(state, _runtime(db_session, user.id))

    normalized = update["normalized_ingredients"]
    assert len(normalized) == 2
    assert all(row.status == IngredientStatus.NEED for row in normalized)


# --- the merge rules (unchanged by CUE-89) ---------------------------------


async def test_duplicate_name_same_unit_merges_and_sums_quantities(
    db_session: AsyncSession, user: User
) -> None:
    ingredients = [
        RecipeIngredient(name="flour", quantity=1, unit="cup"),
        RecipeIngredient(name="pepper", quantity=1, unit="tsp"),
        RecipeIngredient(name="flour", quantity=2, unit="cup"),
    ]

    update = await normalize_ingredients_node(
        _state(ingredients), _runtime(db_session, user.id)
    )

    flour_rows = [
        row for row in update["normalized_ingredients"] if row.name == "flour"
    ]
    assert len(flour_rows) == 1
    assert flour_rows[0].quantity == 3
    assert flour_rows[0].unit == "cup"


async def test_duplicate_name_different_units_kept_separate(
    db_session: AsyncSession, user: User
) -> None:
    ingredients = [
        RecipeIngredient(name="flour", quantity=1, unit="cup"),
        RecipeIngredient(name="flour", quantity=1, unit="g"),
    ]

    update = await normalize_ingredients_node(
        _state(ingredients), _runtime(db_session, user.id)
    )

    flour_rows = [
        row for row in update["normalized_ingredients"] if row.name == "flour"
    ]
    assert len(flour_rows) == 2
    assert {row.unit for row in flour_rows} == {"cup", "g"}
    assert [row.quantity for row in flour_rows] == [1.0, 1.0]


async def test_duplicate_name_both_units_none_kept_separate(
    db_session: AsyncSession, user: User
) -> None:
    ingredients = [
        RecipeIngredient(name="flour", quantity=1, unit=None),
        RecipeIngredient(name="flour", quantity=2, unit=None),
    ]

    update = await normalize_ingredients_node(
        _state(ingredients), _runtime(db_session, user.id)
    )

    flour_rows = [
        row for row in update["normalized_ingredients"] if row.name == "flour"
    ]
    assert len(flour_rows) == 2
    assert all(row.unit is None for row in flour_rows)
    assert sorted(row.quantity for row in flour_rows) == [1.0, 2.0]


async def test_duplicate_name_same_unit_one_quantity_none_merges_to_the_number(
    db_session: AsyncSession, user: User
) -> None:
    ingredients = [
        RecipeIngredient(name="flour", quantity=None, unit="cup"),
        RecipeIngredient(name="flour", quantity=5, unit="cup"),
    ]

    update = await normalize_ingredients_node(
        _state(ingredients), _runtime(db_session, user.id)
    )

    flour_rows = [
        row for row in update["normalized_ingredients"] if row.name == "flour"
    ]
    assert len(flour_rows) == 1
    assert flour_rows[0].quantity == 5


async def test_duplicate_name_same_unit_all_quantities_none_stays_none(
    db_session: AsyncSession, user: User
) -> None:
    ingredients = [
        RecipeIngredient(name="flour", quantity=None, unit="cup"),
        RecipeIngredient(name="flour", quantity=None, unit="cup"),
    ]

    update = await normalize_ingredients_node(
        _state(ingredients), _runtime(db_session, user.id)
    )

    flour_rows = [
        row for row in update["normalized_ingredients"] if row.name == "flour"
    ]
    assert len(flour_rows) == 1
    assert flour_rows[0].quantity is None


async def test_output_order_is_stable_first_appearance(
    db_session: AsyncSession, user: User
) -> None:
    ingredients = [
        RecipeIngredient(name="flour", quantity=1, unit="cup"),
        RecipeIngredient(name="pepper", quantity=1, unit="tsp"),
        RecipeIngredient(name="flour", quantity=1, unit="g"),
        RecipeIngredient(name="sugar", quantity=1, unit="tbsp"),
        RecipeIngredient(name="flour", quantity=1, unit="cup"),
    ]
    state = _state(ingredients)
    runtime = _runtime(db_session, user.id)

    normalized = (await normalize_ingredients_node(state, runtime))[
        "normalized_ingredients"
    ]

    # flour's name is first-seen before pepper and sugar, so all of flour's
    # rows (grouped, unit-ordered by first appearance: tsp then g) come
    # before pepper and sugar.
    assert [(row.name, row.unit) for row in normalized] == [
        ("flour", "cup"),
        ("flour", "g"),
        ("pepper", "tsp"),
        ("sugar", "tbsp"),
    ]

    # Determinism: running again on the same inputs yields an identical result.
    again = await normalize_ingredients_node(state, runtime)
    assert again["normalized_ingredients"] == normalized


# --- contract --------------------------------------------------------------


async def test_raises_on_missing_recipe(db_session: AsyncSession, user: User) -> None:
    state: AgentState = {
        "session_id": "session-1",
        "user_id": 1,
        "messages": [HumanMessage(content="test dish")],
    }

    with pytest.raises(ValueError, match="no recipe"):
        await normalize_ingredients_node(state, _runtime(db_session, user.id))


async def test_returns_partial_update_shape(
    db_session: AsyncSession, user: User
) -> None:
    ingredients = [RecipeIngredient(name="flour", quantity=1, unit="cup")]

    update = await normalize_ingredients_node(
        _state(ingredients), _runtime(db_session, user.id)
    )

    assert set(update.keys()) == {"normalized_ingredients", "have_marks"}
    normalized = update["normalized_ingredients"]
    assert isinstance(normalized, list)
    assert all(isinstance(row, NormalizedIngredient) for row in normalized)
    assert isinstance(update["have_marks"], set)
