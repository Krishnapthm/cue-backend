"""`generate_recipe_node`: dish-name extraction, structured-output retry, and
the domain error surfaced on repeated malformed model output.

These are unit tests - no real model is called. `get_chat_model` is
monkeypatched to a fake `BaseChatModel`-shaped object whose
`with_structured_output(...)` returns a fake runnable, so the suite runs
offline with no provider API key.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from app.agent.config import ModelRole
from app.agent.exceptions import RecipeGenerationError
from app.agent.nodes import recipe as recipe_node
from app.agent.schemas import GeneratedRecipe, RecipeIngredient, RecipeStep
from app.agent.state import AgentState


class _FakeStructuredRunnable:
    """Stands in for `chat_model.with_structured_output(GeneratedRecipe)`.

    Pops one queued result per `ainvoke` call, in order, so a test can queue
    e.g. `[OutputParserException(...), a_valid_recipe]` to exercise the
    retry-then-succeed path.
    """

    def __init__(self, results: list[GeneratedRecipe | Exception]) -> None:
        self._results = list(results)

    async def ainvoke(self, _prompt: list[Any]) -> GeneratedRecipe:
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class _FakeChatModel:
    """Stands in for the `BaseChatModel` returned by `get_chat_model`."""

    def __init__(self, results: list[GeneratedRecipe | Exception]) -> None:
        self._results = results

    def with_structured_output(self, _schema: type) -> _FakeStructuredRunnable:
        return _FakeStructuredRunnable(self._results)


def _state(dish_name: str) -> AgentState:
    return {
        "session_id": "session-1",
        "user_id": 1,
        "messages": [HumanMessage(content=dish_name)],
    }


def _recipe(dish_name: str = "pasta aglio e olio") -> GeneratedRecipe:
    return GeneratedRecipe(
        dish_name=dish_name,
        estimated_time_minutes=20,
        ingredients=[
            RecipeIngredient(name="spaghetti", quantity=200, unit="g"),
            RecipeIngredient(name="garlic", quantity=4, unit="clove"),
            RecipeIngredient(name="olive oil", quantity=60, unit="ml"),
        ],
        method_summary="Boil the pasta, fry sliced garlic in olive oil, toss together.",
        steps=[
            RecipeStep(
                title="Boil the pasta",
                instructions=["Salt the water heavily.", "Cook until al dente."],
                duration_seconds=540,
            ),
            RecipeStep(
                title="Fry the garlic",
                instructions=["Slice the garlic thin.", "Fry gently in the oil."],
            ),
        ],
        servings=2,
        difficulty="Easy",
    )


def _stub_chat_model(
    monkeypatch: pytest.MonkeyPatch, results: list[GeneratedRecipe | Exception]
) -> list[ModelRole]:
    """Stub the model seam, returning the roles the node asked it for."""
    roles: list[ModelRole] = []

    def _get_chat_model(role: ModelRole) -> _FakeChatModel:
        roles.append(role)
        return _FakeChatModel(results)

    monkeypatch.setattr(recipe_node, "get_chat_model", _get_chat_model)
    return roles


async def test_generate_recipe_node_asks_for_the_recipe_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Recipe generation decides correctness, so it gets the strong model -
    # by role, never by naming the model id here.
    roles = _stub_chat_model(monkeypatch, [_recipe()])

    await recipe_node.generate_recipe_node(_state("pasta aglio e olio"))

    assert roles == [ModelRole.RECIPE]


async def test_generate_recipe_node_returns_recipe_on_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe = _recipe()
    _stub_chat_model(monkeypatch, [recipe])

    update = await recipe_node.generate_recipe_node(_state("pasta aglio e olio"))

    # The node returns a partial state update, not a full AgentState.
    assert update["recipe"] == recipe
    assert update["recipe"].estimated_time_minutes == 20
    assert [i.name for i in update["recipe"].ingredients] == [
        "spaghetti",
        "garlic",
        "olive oil",
    ]


async def test_generate_recipe_node_obscure_dish_still_returns_recipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The model itself is responsible for the best-effort behaviour (prompt
    # design); here we just confirm the node doesn't special-case or reject
    # an unfamiliar dish name - whatever GeneratedRecipe the model produces
    # flows straight onto state.
    coarse_recipe = GeneratedRecipe(
        dish_name="grandma's secret regional stew",
        estimated_time_minutes=45,
        ingredients=[RecipeIngredient(name="assorted vegetables")],
        method_summary="Simmer everything together until tender.",
        steps=[
            RecipeStep(
                title="Simmer everything",
                instructions=["Cover and simmer until tender."],
                duration_seconds=2700,
            )
        ],
    )
    _stub_chat_model(monkeypatch, [coarse_recipe])

    update = await recipe_node.generate_recipe_node(
        _state("grandma's secret regional stew")
    )

    assert update["recipe"].dish_name == "grandma's secret regional stew"
    assert update["recipe"].ingredients[0].quantity is None


async def test_generate_recipe_node_retries_once_on_malformed_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe = _recipe()
    _stub_chat_model(
        monkeypatch, [OutputParserException("could not parse model output"), recipe]
    )

    update = await recipe_node.generate_recipe_node(_state("pasta aglio e olio"))

    assert update["recipe"] == recipe


async def test_generate_recipe_node_raises_domain_error_after_two_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_chat_model(
        monkeypatch,
        [
            OutputParserException("could not parse model output"),
            OutputParserException("still could not parse model output"),
        ],
    )

    with pytest.raises(RecipeGenerationError):
        await recipe_node.generate_recipe_node(_state("pasta aglio e olio"))


async def test_generate_recipe_node_raises_on_empty_messages() -> None:
    empty_state: AgentState = {"session_id": "session-1", "user_id": 1, "messages": []}

    with pytest.raises(ValueError, match="no messages"):
        await recipe_node.generate_recipe_node(empty_state)


async def test_generate_recipe_node_defers_the_rendered_reply_until_after_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Rendering happens after the ready-made choice has either been skipped or
    # answered, so the user never sees constituents before choosing them.
    recipe = _recipe()
    _stub_chat_model(monkeypatch, [recipe])

    update = await recipe_node.generate_recipe_node(_state("pasta aglio e olio"))

    assert "messages" not in update


def test_render_recipe_lists_every_ingredient() -> None:
    rendered = recipe_node.render_recipe(_recipe())

    assert "pasta aglio e olio" in rendered
    assert "20 minutes" in rendered
    assert "- spaghetti - 200 g" in rendered
    assert "- garlic - 4 clove" in rendered
    assert "- olive oil - 60 ml" in rendered
    assert "Boil the pasta" in rendered


def test_render_recipe_omits_a_missing_quantity() -> None:
    recipe = GeneratedRecipe(
        dish_name="salt water",
        estimated_time_minutes=1,
        ingredients=[
            RecipeIngredient(name="salt"),
            RecipeIngredient(name="water", quantity=1, unit="l"),
            # A quantity with no unit still renders the number.
            RecipeIngredient(name="ice", quantity=3),
        ],
        method_summary="Dissolve.",
        steps=[RecipeStep(title="Dissolve", instructions=["Stir until dissolved."])],
    )

    rendered = recipe_node.render_recipe(recipe)

    assert "- salt\n" in rendered
    assert "- water - 1 l" in rendered
    assert "- ice - 3" in rendered


def test_render_recipe_formats_fractional_quantities() -> None:
    recipe = GeneratedRecipe(
        dish_name="dressing",
        estimated_time_minutes=2,
        ingredients=[RecipeIngredient(name="vinegar", quantity=1.5, unit="tbsp")],
        method_summary="Whisk.",
        steps=[RecipeStep(title="Whisk", instructions=["Whisk it."])],
    )

    # Whole numbers lose the .0, but a real fraction is preserved.
    assert "- vinegar - 1.5 tbsp" in recipe_node.render_recipe(recipe)


def test_render_recipe_handles_an_empty_ingredient_list() -> None:
    recipe = GeneratedRecipe(
        dish_name="unrecognized",
        estimated_time_minutes=0,
        ingredients=[],
        method_summary="Nothing recipe-related was recognized.",
        steps=[
            RecipeStep(
                title="Nothing recognized",
                instructions=["Nothing recipe-related was recognized."],
            )
        ],
    )

    rendered = recipe_node.render_recipe(recipe)

    assert "No ingredients were identified" in rendered
    assert "Ingredients:" not in rendered


def test_generated_recipe_requires_at_least_one_step() -> None:
    # `steps` is the whole point of CUE-116, and both features downstream (the
    # reveal card, cooking mode) have nothing to render without it. A model
    # that omits it is a malformed structured output, which the node already
    # knows how to retry - so the floor belongs in the schema, not in a
    # defensive check at every reader.
    with pytest.raises(ValidationError):
        GeneratedRecipe(
            dish_name="pasta",
            estimated_time_minutes=20,
            ingredients=[],
            method_summary="Boil.",
            steps=[],
        )


def test_recipe_step_requires_at_least_one_instruction_line() -> None:
    with pytest.raises(ValidationError):
        RecipeStep(title="Soften the onions", instructions=[])


def test_recipe_step_rejects_a_zero_second_timer() -> None:
    # A zero-second timer is not a timer; `None` is how an untimed step says so.
    with pytest.raises(ValidationError):
        RecipeStep(title="Chop", instructions=["Chop the onions."], duration_seconds=0)


async def test_generate_recipe_node_carries_steps_onto_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe = _recipe()
    _stub_chat_model(monkeypatch, [recipe])

    update = await recipe_node.generate_recipe_node(_state("pasta aglio e olio"))

    steps = update["recipe"].steps
    assert [step.title for step in steps] == ["Boil the pasta", "Fry the garlic"]
    assert all(step.instructions for step in steps)
    # A timed step keeps its duration; an untimed one stays null rather than
    # guessing a number cooking mode would then count down.
    assert steps[0].duration_seconds == 540
    assert steps[1].duration_seconds is None


async def test_generate_recipe_node_carries_the_cards_meta_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_chat_model(monkeypatch, [_recipe()])

    update = await recipe_node.generate_recipe_node(_state("pasta aglio e olio"))

    assert update["recipe"].servings == 2
    assert update["recipe"].difficulty == "Easy"


async def test_generate_recipe_node_accepts_a_recipe_with_no_meta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `servings` and `difficulty` are absent-safe: the model omits either
    # rather than inventing one, and nothing downstream requires them.
    recipe = GeneratedRecipe(
        dish_name="toast",
        estimated_time_minutes=3,
        ingredients=[RecipeIngredient(name="bread", quantity=2, unit="slice")],
        method_summary="Toast the bread.",
        steps=[RecipeStep(title="Toast", instructions=["Toast the bread."])],
    )
    _stub_chat_model(monkeypatch, [recipe])

    update = await recipe_node.generate_recipe_node(_state("toast"))

    assert update["recipe"].servings is None
    assert update["recipe"].difficulty is None


def test_render_recipe_still_shows_only_the_summary() -> None:
    # `render_recipe` is unchanged by CUE-116 on purpose: steps reach the user
    # through the recipe card (CUE-118), never through the chat text, so the
    # reply must not start duplicating them.
    rendered = recipe_node.render_recipe(_recipe())

    assert "Boil the pasta, fry sliced garlic" in rendered
    assert "Fry the garlic" not in rendered
    assert "Salt the water heavily" not in rendered


def test_both_prompts_ask_for_steps_and_a_conditional_timer() -> None:
    for prompt in (recipe_node._SYSTEM_PROMPT, recipe_node._PHOTO_SYSTEM_PROMPT):
        assert "steps" in prompt
        assert "duration_seconds" in prompt
        # The timer is opt-in per step, and the prompt has to say so - a model
        # that guesses a duration on every step is the failure this rules out.
        assert "ONLY" in prompt
        assert "null" in prompt


def test_the_photo_prompt_keeps_one_step_when_the_photo_is_not_a_recipe() -> None:
    # `steps` stays min_length=1 on the photo path, so the "not a recipe at
    # all" branch has to be told to emit one explanatory step rather than none.
    prompt = recipe_node._PHOTO_SYSTEM_PROMPT

    assert "exactly ONE step" in prompt
    assert "Never return an empty steps list" in prompt
