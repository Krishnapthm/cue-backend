from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any, cast

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ImageContentBlock,
    SystemMessage,
    TextContentBlock,
)
from pydantic import ValidationError

from app.agent.config import ModelRole
from app.agent.exceptions import RecipeGenerationError
from app.agent.providers import get_chat_model
from app.agent.schemas import GeneratedRecipe, RecipeIngredient
from app.agent.state import AgentState
from app.agent.storage import get_image_store

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


def _render_ingredient(ingredient: RecipeIngredient) -> str:
    """Render one ingredient line as `name - quantity unit`.

    Quantity and unit are each optional, so the amount is only appended when
    there is one. A whole-number quantity renders without a trailing `.0`,
    since "2 clove" reads like a recipe and "2.0 clove" does not.
    """
    if ingredient.quantity is None:
        return f"- {ingredient.name}"

    quantity = ingredient.quantity
    amount = f"{quantity:g}"
    if ingredient.unit:
        amount = f"{amount} {ingredient.unit}"
    return f"- {ingredient.name} - {amount}"


def render_recipe(recipe: GeneratedRecipe) -> str:
    """Render a validated recipe as the assistant's chat reply.

    Deterministic formatting of fields that already passed `GeneratedRecipe`
    validation - deliberately not a second model call. The reply is the
    ingredient list (the product's core output), with the method summary as
    a closing line.

    Args:
        recipe: The structured recipe to render.

    Returns:
        Plain text suitable for an `AIMessage` body.
    """
    lines = [f"{recipe.dish_name} (about {recipe.estimated_time_minutes} minutes)", ""]
    if recipe.ingredients:
        lines.append("Ingredients:")
        lines.extend(_render_ingredient(i) for i in recipe.ingredients)
    else:
        lines.append("No ingredients were identified for this dish.")
    lines.extend(["", recipe.method_summary])
    return "\n".join(lines)


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
        A partial state update containing the generated `recipe` and the
        `AIMessage` rendering it for the transcript. The message is
        deterministic formatting of already-validated fields, not a second
        model call. This is a partial dict rather than a full `AgentState`
        (LangGraph's node convention) - a partial dict cannot satisfy the
        total `AgentState` TypedDict under strict typing. `messages` is
        appended, not overwritten, by the `add_messages` reducer.

        `have_marks` is cleared, because a new recipe means a new checklist.
        The field survives in the checkpoint across turns, and
        `normalize_ingredients_node` treats a non-empty `have_marks` as the
        user's own answer and skips the pantry seed on that basis - so leaving
        the *previous* turn's marks in place would suppress seeding on every
        turn after the first, and let a mark meant for one dish decide an
        identically named ingredient in another.

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

    structured_model = get_chat_model(ModelRole.RECIPE).with_structured_output(
        GeneratedRecipe
    )
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

    return {
        "recipe": recipe,
        "messages": [AIMessage(content=render_recipe(recipe))],
        "have_marks": set(),
    }


_PHOTO_SYSTEM_PROMPT = (
    "You are a recipe generation assistant. You will be given a photo of a "
    "recipe (e.g. a cookbook page, a handwritten note, a packaging label, or "
    "a screenshot). Read the image and produce the same structured recipe "
    "schema used for text-based recipe generation: a list of ingredients "
    "(name, quantity, unit) and a brief method summary.\n"
    "\n"
    "Rules:\n"
    "- Always produce a complete, best-effort structured recipe, even if the "
    "photo is blurry, partially cropped, low-resolution, or otherwise hard "
    "to read. Never refuse and never respond with an error message in place "
    "of the structured output - extract everything you can read with "
    "reasonable confidence.\n"
    "- If the photo is not a recipe at all (e.g. an unrelated photo with no "
    "ingredients or cooking instructions visible), return an EMPTY "
    "ingredients list and set method_summary to a brief plain-text note that "
    "nothing recipe-related was recognized in the image. Do this instead of "
    "raising an error or refusing.\n"
    "- Quantities and units are optional per ingredient - omit them (leave "
    "null) rather than guessing a number you cannot read with reasonable "
    "confidence.\n"
    "- The method summary must be brief plain text (a few sentences), never "
    "markdown, bullet points, or numbered steps.\n"
    "- estimated_time_minutes is your best-effort total time (prep + cook) "
    "in minutes; if the image is unreadable or not a recipe, use 0.\n"
    "- dish_name is your best-effort read of the dish's name; if unreadable "
    "or not a recipe at all, use a short literal description of what the "
    "image shows instead."
)

_IMAGE_MIME_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".heic": "image/heic",
    ".heif": "image/heif",
}
_DEFAULT_IMAGE_MIME_TYPE = "image/jpeg"


def _infer_image_mime_type(object_path: str) -> str:
    """Infer an image MIME type from an object path's suffix.

    Args:
        object_path: The Supabase Storage object path (e.g.
            "recipes/abc123.jpg").

    Returns:
        The inferred MIME type, or `_DEFAULT_IMAGE_MIME_TYPE` if the suffix
        is unrecognized or absent.
    """
    suffix = Path(object_path).suffix.lower()
    return _IMAGE_MIME_TYPES.get(suffix, _DEFAULT_IMAGE_MIME_TYPE)


async def parse_recipe_photo_node(state: AgentState) -> dict[str, Any]:
    """Parse an uploaded recipe photo into a structured recipe.

    Reads the Supabase Storage object path from
    `state["image_object_path"]` - carried from `ChatMessage.payload`
    (kind='image', see `app/models/chat.py`); the payload -> state extraction
    at the app/graph boundary happens in a later wiring issue, so it is out
    of scope here (this node, like `generate_recipe_node`, is not yet wired
    into `build_graph`). Nodes only receive `state`, never a DB session, so
    the image bytes are fetched through the `get_image_store` seam
    (`app.agent.storage`) rather than a repository call.

    The fetched bytes are base64-encoded into a provider-agnostic image
    content block and sent to the configured chat model with the same
    `with_structured_output(GeneratedRecipe)` contract `generate_recipe_node`
    uses, so both intake paths (typed dish name, uploaded photo) render
    through the exact same `GeneratedRecipe` schema - one review surface, per
    the issue's acceptance criteria.

    Downscaling large source images is deliberately NOT done here: per the
    issue, an oversized source image is a cost/latency concern, not a
    correctness one, so resizing belongs at the upload boundary (before the
    object ever reaches storage), not in this node. Adding an image library
    here to shrink pixels would be scope creep against that framing.

    Args:
        state: The current graph state. Must have `image_object_path` set to
            a non-empty Supabase Storage object path.

    Returns:
        A partial state update containing the parsed `recipe`, and a cleared
        `have_marks` for the reason `generate_recipe_node`'s docstring gives.
        This is a partial dict rather than a full `AgentState` (see that
        docstring for why).

    Raises:
        ValueError: `state["image_object_path"]` is missing or empty, so
            there is no image to parse.
        RecipeGenerationError: The model failed to produce a valid
            structured recipe twice in a row (initial attempt + one retry).
    """
    object_path = state.get("image_object_path")
    if not object_path:
        raise ValueError(
            "Cannot parse a recipe photo: state has no image_object_path to "
            "load an image from."
        )

    image_bytes = await get_image_store().load(object_path)
    image_base64 = base64.b64encode(image_bytes).decode("ascii")
    mime_type = _infer_image_mime_type(object_path)

    structured_model = get_chat_model(ModelRole.VISION).with_structured_output(
        GeneratedRecipe
    )
    # `HumanMessage.content` is typed as `list[str | dict[str, Any]]`; the
    # standard content-block TypedDicts are structurally dicts at runtime, so
    # the cast below only satisfies mypy - it does not change the payload.
    content: list[str | dict[str, Any]] = [
        cast(
            "dict[str, Any]",
            TextContentBlock(
                type="text", text="Extract the recipe shown in this photo."
            ),
        ),
        cast(
            "dict[str, Any]",
            ImageContentBlock(type="image", base64=image_base64, mime_type=mime_type),
        ),
    ]
    prompt = [
        SystemMessage(content=_PHOTO_SYSTEM_PROMPT),
        HumanMessage(content=content),
    ]

    try:
        recipe = await structured_model.ainvoke(prompt)
    except (ValidationError, OutputParserException) as exc:
        logger.warning(
            "Recipe photo parse returned malformed output for %r; retrying once: %s",
            object_path,
            exc,
        )
        try:
            recipe = await structured_model.ainvoke(prompt)
        except (ValidationError, OutputParserException) as retry_exc:
            logger.error(
                "Recipe photo parse failed again for %r after retry: %s",
                object_path,
                retry_exc,
            )
            raise RecipeGenerationError() from retry_exc

    if not isinstance(recipe, GeneratedRecipe):
        # with_structured_output(GeneratedRecipe) is documented to return a
        # GeneratedRecipe instance (not a raw dict) when passed a Pydantic
        # schema; this guards that contract defensively at runtime.
        raise RecipeGenerationError()

    return {"recipe": recipe, "have_marks": set()}
