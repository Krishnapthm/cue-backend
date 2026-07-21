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
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.exceptions import RecipeGenerationError
from app.agent.nodes import recipe as recipe_node
from app.agent.schemas import GeneratedRecipe, RecipeIngredient
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
    )


def _stub_chat_model(
    monkeypatch: pytest.MonkeyPatch, results: list[GeneratedRecipe | Exception]
) -> None:
    monkeypatch.setattr(recipe_node, "get_chat_model", lambda: _FakeChatModel(results))


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


async def test_generate_recipe_node_appends_a_rendered_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The reply is deterministic formatting of already-validated fields, not
    # a second model call - the fake queues exactly one result, so a second
    # call would raise IndexError.
    recipe = _recipe()
    _stub_chat_model(monkeypatch, [recipe])

    update = await recipe_node.generate_recipe_node(_state("pasta aglio e olio"))

    messages = update["messages"]
    assert len(messages) == 1
    assert isinstance(messages[0], AIMessage)
    assert str(messages[0].content) == recipe_node.render_recipe(recipe)


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
    )

    # Whole numbers lose the .0, but a real fraction is preserved.
    assert "- vinegar - 1.5 tbsp" in recipe_node.render_recipe(recipe)


def test_render_recipe_handles_an_empty_ingredient_list() -> None:
    recipe = GeneratedRecipe(
        dish_name="unrecognized",
        estimated_time_minutes=0,
        ingredients=[],
        method_summary="Nothing recipe-related was recognized.",
    )

    rendered = recipe_node.render_recipe(recipe)

    assert "No ingredients were identified" in rendered
    assert "Ingredients:" not in rendered
