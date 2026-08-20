from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any, cast

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import (
    HumanMessage,
    ImageContentBlock,
    SystemMessage,
    TextContentBlock,
)
from pydantic import ValidationError

from app.agent.config import ModelRole
from app.agent.exceptions import RecipeGenerationError
from app.agent.providers import get_chat_model
from app.agent.schemas import GeneratedRecipe, RecipeIngredient, TurnIntent
from app.agent.state import AgentState
from app.agent.storage import get_image_store

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a recipe generation assistant. Given a dish name, produce a "
    "structured recipe: a list of ingredients (name, quantity, unit), a "
    "brief method summary, and the cooking steps the user will actually "
    "follow.\n"
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
    "- Populate scratch_components only for substantial sub-components that "
    "have at least two named ingredients in this recipe and are commonly sold "
    "ready-made (for example dosa batter, samosa pastry, pizza dough, curry "
    "paste, stock, or a ground spice mix). For each, give its user-facing "
    "name, the ready-made item to search for, and the exact ingredient names "
    "it replaces. Leave it empty when there is no meaningful choice.\n"
    "- Quantities and units are optional per ingredient - omit them (leave "
    "null) rather than guessing a number you are not reasonably confident "
    "about.\n"
    "- The method summary must be brief plain text (a few sentences), never "
    "markdown, bullet points, or numbered steps - it is a summary, not a "
    "full recipe card.\n"
    "- estimated_time_minutes is your best-effort total time (prep + cook) "
    "in minutes.\n"
    "- steps is the method the user cooks from, in cooking order. Give at "
    "least one step. Each step has a short imperative title ('Soften the "
    "onions'), 1-4 short imperative instruction lines, and nothing else - "
    "no step numbers in the title, no markdown.\n"
    "- Set duration_seconds on a step ONLY where it has a genuine timed or "
    "unattended stretch: a simmer, a rest, a proof, a bake, a marinade. "
    "Leave it null on every other step. Do not guess a number for chopping, "
    "mixing, or plating - a wrong timer is worse than no timer.\n"
    "- steps and method_summary are not the same thing and both are "
    "required: the summary stays a brief prose overview, the steps are the "
    "instructions. Do not put the summary in the steps or vice versa.\n"
    "- servings is the number of people the quantities above serve, and "
    "difficulty is one short label - exactly 'Easy', 'Medium', or 'Hard'. "
    "Omit either (leave null) rather than guessing one you are not "
    "reasonably confident about."
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


def _render_parsed_photo(state: AgentState) -> dict[str, Any]:
    """Hand a photo-parsed recipe on to the checklist, unchanged (CUE-88).

    `parse_recipe_photo -> generate_recipe` is what lets a photo turn rejoin the
    normal path and get a checklist and a cart like any other. What it must
    *not* do is generate a second recipe: the photo has already been read, and
    the latest message on a photo turn is a caption or nothing at all. So this
    branch spends no model call and renders the parsed recipe for the
    transcript - the reply `parse_recipe_photo_node` deliberately does not emit.

    Gated on `turn_intent`, not on "a recipe is already in state": `recipe`
    survives in the checkpoint across turns, so the second text turn of a
    session always arrives with the first turn's recipe still set. Keying on
    the intent of *this* turn is what stops that from silently swallowing a new
    dish name.

    Args:
        state: The current graph state, on a turn the router labelled PHOTO.

    Returns:
        A partial state update. `recipe` is already on state and is not
        rewritten; rendering waits until the scratch-choice step.

    Raises:
        RecipeGenerationError: The turn is a photo turn but carries no parsed
            recipe. That means `parse_recipe_photo_node` did not run or did not
            store its result, which is a wiring bug - and falling back to
            generating from the caption would silently answer with a different
            dish than the one in the photo.
    """
    recipe = state.get("recipe")
    if recipe is None:
        logger.error(
            "Photo turn for session %s reached generate_recipe with no parsed "
            "recipe; refusing to generate one from the caption.",
            state["session_id"],
        )
        raise RecipeGenerationError()
    return {"scratch_component": None, "scratch_choice": None}


async def generate_recipe_node(state: AgentState) -> dict[str, Any]:
    """Generate a structured recipe for the dish named in the latest message.

    On a turn the router labelled `PHOTO` this generates nothing: the recipe was
    already read off the image upstream, and this node only renders it into the
    transcript. See `_render_parsed_photo`.

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
        A partial state update containing the generated `recipe`. Rendering
        waits until the pre-checklist scratch-choice step, so ingredients do
        not appear before the user has made a meaningful source choice. This
        is a partial dict rather than a full `AgentState`
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
    if state.get("turn_intent") is TurnIntent.PHOTO:
        return _render_parsed_photo(state)

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
        "have_marks": set(),
        "scratch_component": None,
        "scratch_choice": None,
    }


_PHOTO_SYSTEM_PROMPT = (
    "You are a recipe generation assistant. You will be given a photo of a "
    "recipe (e.g. a cookbook page, a handwritten note, a packaging label, or "
    "a screenshot). Read the image and produce the same structured recipe "
    "schema used for text-based recipe generation: a list of ingredients "
    "(name, quantity, unit), a brief method summary, and the cooking "
    "steps.\n"
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
    "raising an error or refusing. steps must still hold exactly ONE step, "
    "whose title and single instruction line say the same thing in plain "
    "text - that nothing recipe-related was recognized. Never return an "
    "empty steps list.\n"
    "- Quantities and units are optional per ingredient - omit them (leave "
    "null) rather than guessing a number you cannot read with reasonable "
    "confidence.\n"
    "- The method summary must be brief plain text (a few sentences), never "
    "markdown, bullet points, or numbered steps.\n"
    "- estimated_time_minutes is your best-effort total time (prep + cook) "
    "in minutes; if the image is unreadable or not a recipe, use 0.\n"
    "- dish_name is your best-effort read of the dish's name; if unreadable "
    "or not a recipe at all, use a short literal description of what the "
    "image shows instead.\n"
    "- steps is the method the user cooks from, in cooking order, read off "
    "the image. Give at least one step. Each step has a short imperative "
    "title ('Soften the onions'), 1-4 short imperative instruction lines, "
    "and nothing else - no step numbers in the title, no markdown. If the "
    "photo shows ingredients but no method, infer the minimum plausible "
    "steps rather than returning none.\n"
    "- Set duration_seconds on a step ONLY where the image states or clearly "
    "implies a timed or unattended stretch: a simmer, a rest, a proof, a "
    "bake, a marinade. Leave it null on every other step, and never guess a "
    "number you cannot read - a wrong timer is worse than no timer.\n"
    "- steps and method_summary are not the same thing and both are "
    "required: the summary stays a brief prose overview, the steps are the "
    "instructions. Do not put the summary in the steps or vice versa.\n"
    "- servings is the number of people the recipe serves and difficulty is "
    "one short label - exactly 'Easy', 'Medium', or 'Hard'. Omit either "
    "(leave null) rather than guessing one the image does not support."
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

    That includes `steps`, which stays `min_length=1` on this path rather than
    being relaxed to `min_length=0` for the "not a recipe at all" branch. A
    photo with nothing recipe-related in it returns an empty `ingredients` list
    and a *single explanatory step* saying so, which the prompt asks for
    explicitly. One invariant ("a recipe always has at least one step") is
    worth more than the branch it saves here: relaxing the floor would push an
    empty-list case onto the reveal card and cooking mode, neither of which has
    anything to render for it.

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

    return {
        "recipe": recipe,
        "have_marks": set(),
        "scratch_component": None,
        "scratch_choice": None,
    }
