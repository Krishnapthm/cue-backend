from __future__ import annotations

import logging
from typing import Any

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from app.agent.exceptions import RecipeGenerationError
from app.agent.providers import get_chat_model
from app.agent.schemas import GeneratedRecipe
from app.agent.state import AgentState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a recipe generation assistant. Given a dish name, produce a "
    "structured recipe: a list of ingredients (name, quantity, unit) and a "
    "brief method summary.\n"
    "\n"
    "Rules:\n"
    "- Always produce a complete, best-effort recipe, even if the dish name "
    "is obscure, regional, misspelled, or you are not fully certain what it "
    "refers to. Never refuse and never ask a clarifying question - infer the "
    "most plausible dish and generate accordingly.\n"
    "- If the dish is unfamiliar, use your best judgement to approximate a "
    "reasonable recipe and make the method summary coarser/more general "
    "rather than omitting it.\n"
    "- List every ingredient the dish plausibly needs; do not truncate or "
    "cap the ingredient list for the sake of brevity.\n"
    "- Quantities and units are optional per ingredient - omit them (leave "
    "null) rather than guessing a number you are not reasonably confident "
    "about.\n"
    "- The method summary must be brief plain text (a few sentences), never "
    "markdown, bullet points, or numbered steps - it is a summary, not a "
    "full recipe card.\n"
    "- estimated_time_minutes is your best-effort total time (prep + cook) "
    "in minutes."
)


async def generate_recipe_node(state: AgentState) -> dict[str, Any]:
    """Generate a structured recipe for the dish named in the latest message.

    Extracts the dish name from `state["messages"][-1].content` (the latest
    user message) and calls the configured chat model with a structured-
    output prompt so the model returns a `GeneratedRecipe` directly rather
    than free text to be parsed - the tight schema is the hallucination
    backstop per R1.2. On a malformed structured output (a Pydantic
    validation failure), the call is retried once with the same prompt before
    raising `RecipeGenerationError`.

    Args:
        state: The current graph state. Must have at least one message; the
            last one is treated as the user's dish-name intent.

    Returns:
        A partial state update containing the generated `recipe`. This is a
        partial dict rather than a full `AgentState` (LangGraph's node
        convention, see `smoke_test_node` in `app/agent/graph.py`) - a
        partial dict cannot satisfy the total `AgentState` TypedDict under
        strict typing.

    Raises:
        ValueError: `state["messages"]` is empty, so there is no dish name to
            extract.
        RecipeGenerationError: The model failed to produce a valid
            structured recipe twice in a row (initial attempt + one retry).
    """
    messages = state["messages"]
    if not messages:
        raise ValueError(
            "Cannot generate a recipe: state has no messages to extract a "
            "dish name from."
        )
    dish_name = str(messages[-1].content)

    structured_model = get_chat_model().with_structured_output(GeneratedRecipe)
    prompt = [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=dish_name)]

    try:
        recipe = await structured_model.ainvoke(prompt)
    except (ValidationError, OutputParserException) as exc:
        logger.warning(
            "Recipe generation returned malformed output for dish %r; "
            "retrying once: %s",
            dish_name,
            exc,
        )
        try:
            recipe = await structured_model.ainvoke(prompt)
        except (ValidationError, OutputParserException) as retry_exc:
            logger.error(
                "Recipe generation failed again for dish %r after retry: %s",
                dish_name,
                retry_exc,
            )
            raise RecipeGenerationError() from retry_exc

    if not isinstance(recipe, GeneratedRecipe):
        # with_structured_output(GeneratedRecipe) is documented to return a
        # GeneratedRecipe instance (not a raw dict) when passed a Pydantic
        # schema; this guards that contract defensively at runtime.
        raise RecipeGenerationError()

    return {"recipe": recipe}
