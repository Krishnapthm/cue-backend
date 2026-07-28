"""Reshape a generated recipe into the have/need checklist (CUE-24, CUE-89).

CUE-24 built the reshaping; CUE-89 wired the node into the graph and gave it
the one behaviour that makes the checklist worth showing - the user's pantry
seeds it, so the staples they already keep arrive already ticked instead of
being re-ticked by hand on every single turn.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.runtime import Runtime

from app.agent.context import CueContext
from app.agent.schemas import IngredientStatus, NormalizedIngredient
from app.agent.state import AgentState
from app.pantry import service as pantry_service
from app.pantry.service import normalize_name

logger = logging.getLogger(__name__)

#: Ingredients omitted from every checklist because they are universally present.
#: Keep this deliberately narrow: omitting an ingredient a user needs to buy is
#: worse than leaving an extra row for them to confirm.
ASSUMED_STAPLES = frozenset({"salt", "water"})

#: A bulk mass amount is an explicit exception to `ASSUMED_STAPLES`. A recipe
#: calling for this much salt or water may be making a salt crust or another
#: non-pantry preparation, so retaining it is safer than silently excluding it.
_BULK_STAPLE_MASS_GRAMS = 500
_GRAM_UNITS = frozenset({"g", "gram", "grams"})
_KILOGRAM_UNITS = frozenset({"kg", "kilogram", "kilograms"})


def _is_bulk_staple(quantity: float | None, unit: str | None) -> bool:
    """Return whether an assumed staple has an unusually large mass amount."""
    if quantity is None or unit is None:
        return False

    normalized_unit = normalize_name(unit)
    if normalized_unit in _GRAM_UNITS:
        return quantity >= _BULK_STAPLE_MASS_GRAMS
    return (
        normalized_unit in _KILOGRAM_UNITS
        and quantity * 1000 >= _BULK_STAPLE_MASS_GRAMS
    )


def _is_assumed_staple(name: str, quantity: float | None, unit: str | None) -> bool:
    """Return whether an ingredient can safely be omitted from a checklist."""
    return normalize_name(name) in ASSUMED_STAPLES and not _is_bulk_staple(
        quantity, unit
    )


async def _seed_have_marks(
    recipe_names: list[str], runtime: Runtime[CueContext]
) -> set[str]:
    """Pre-mark the recipe's ingredients the user's pantry says they have.

    Matching is exact on `pantry.service.normalize_name`'s form, which is the
    codebase's single definition of "the same item" - the same rule
    `stamp_last_bought` already matches ingredient names to staples with. A
    second, looser scheme here (substrings, stemming, edit distance) would be a
    second definition of the same thing, free to disagree with the first.

    Exact-on-normalized deliberately **under**-matches: "rice" does not match
    the staple "basmati rice". That is the right direction to be wrong in. A
    missed match costs the user one tick; a false HAVE silently drops an
    ingredient from the order and they find out at the stove.

    Args:
        recipe_names: The recipe's ingredient names, as the recipe spells them.
        runtime: The turn's runtime context, for the request's session and user.

    Returns:
        The subset of `recipe_names` that matched an in-stock staple, spelled
        as the *recipe* spells them - `have_marks` is keyed on ingredient
        names, so the pantry's spelling would never match anything downstream.
    """
    stocked = await pantry_service.stocked_names(
        runtime.context.session, runtime.context.user_id
    )
    if not stocked:
        return set()
    return {name for name in recipe_names if normalize_name(name) in stocked}


async def normalize_ingredients_node(
    state: AgentState, runtime: Runtime[CueContext]
) -> dict[str, Any]:
    """Reshape `GeneratedRecipe.ingredients` into structured have/need rows.

    Deterministic, not model-driven: maps `state["recipe"].ingredients` (from
    either intake path - CUE-22 dish-name generation or CUE-23 photo parse,
    both produce a `GeneratedRecipe`) plus the have-list checkbox state in
    `state["have_marks"]` into `list[NormalizedIngredient]`. Ingredients in
    `ASSUMED_STAPLES` are removed before any checklist or cart path sees them,
    except for explicitly bulk mass quantities. Every remaining source
    ingredient produces a row -
    `HAVE` rows are kept for display, and only `NEED` rows are later handed
    to matching (the caller filters on `status`, not this node).

    Merge rule for duplicate ingredient names (documented here so a reviewer
    can check intent against behaviour):
      - Ingredients are grouped by exact name match, in first-appearance
        order of the name.
      - Within a name group, rows are sub-grouped by unit, in first-
        appearance order of the unit within that group.
      - Two rows with the SAME non-None unit merge into a single row: the
        merged quantity is the sum of the non-None quantities among them: if
        every contributing quantity is None the merged quantity stays None,
        otherwise the None entries are ignored and the numbers are summed.
      - Rows with a missing (`None`) unit never merge - not with each other
        and not with a unit-bearing row - since "units differ or are
        missing" treats a missing unit as always distinct. Each such row
        stays its own entry.
      - `status` is HAVE if the ingredient's name is in `have_marks`, else
        NEED; this applies uniformly to every row produced for that name.
        A `have_marks` entry that does not exact-match any ingredient name
        is silently ignored (it just never marks anything) - it never
        raises.

    Where `have_marks` comes from, and who wins:

      - The user's marks are authoritative. If `have_marks` already carries
        values - a resume of `confirm_checklist`, or an explicit client
        submission - they are used exactly as given and the pantry is not even
        read. Pantry state is a suggestion about what is in the cupboard; the
        user's answer is the fact, and re-applying the suggestion on top of it
        would silently re-tick a staple the user had just unticked because they
        ran out.
      - Otherwise the marks are seeded from the pantry, so the obvious things
        arrive already ticked. See `_seed_have_marks` for the matching rule and
        why it errs towards NEED.

    The seeded set is returned in the state update, not just used locally:
    `confirm_checklist` (CUE-90) interrupts with this checklist and resumes by
    overwriting the same field, so the seed has to be the visible starting
    value of the thing the user is about to edit.

    Args:
        state: The current graph state. Must have `recipe` set to a
            `GeneratedRecipe`; `have_marks` is optional and is seeded from the
            pantry when absent or empty.
        runtime: The turn's runtime context, carrying the request's database
            session and user. The pantry read goes through it rather than
            through state, because an `AsyncSession` can never be checkpointed
            - see `app/agent/context.py`.

    Returns:
        A partial state update containing `normalized_ingredients` and the
        `have_marks` they were derived from. This is a partial dict rather
        than a full `AgentState` (LangGraph's node convention) - a partial
        dict cannot satisfy the total `AgentState` TypedDict under strict
        typing.

    Raises:
        ValueError: `state["recipe"]` is missing or `None`, so there is
            nothing to normalize.
    """
    recipe = state.get("recipe")
    if recipe is None:
        raise ValueError(
            "Cannot normalize ingredients: state has no recipe to normalize."
        )

    recipe_ingredients = [
        ingredient
        for ingredient in recipe.ingredients
        if not _is_assumed_staple(ingredient.name, ingredient.quantity, ingredient.unit)
    ]

    have_marks = state.get("have_marks") or set()
    if have_marks:
        logger.debug(
            "Session %s supplied %d have-marks; skipping the pantry seed.",
            state["session_id"],
            len(have_marks),
        )
    else:
        have_marks = await _seed_have_marks(
            [ingredient.name for ingredient in recipe_ingredients], runtime
        )
        logger.debug(
            "Seeded %d have-marks from the pantry for session %s.",
            len(have_marks),
            state["session_id"],
        )

    # name -> ordered list of unit buckets; each bucket accumulates the
    # quantities of every source row sharing that (name, unit) pair. Buckets
    # with unit=None are never reused - each None-unit row gets its own
    # bucket - so they never merge with one another.
    buckets_by_name: dict[str, list[dict[str, Any]]] = {}
    # name -> unit -> index of that unit's bucket in buckets_by_name[name],
    # used to find the existing bucket to merge a repeated non-None unit into.
    unit_bucket_index: dict[str, dict[str, int]] = {}

    for ingredient in recipe_ingredients:
        name = ingredient.name
        buckets = buckets_by_name.setdefault(name, [])
        if ingredient.unit is None:
            buckets.append({"unit": None, "quantities": [ingredient.quantity]})
        else:
            unit_index = unit_bucket_index.setdefault(name, {})
            existing_index = unit_index.get(ingredient.unit)
            if existing_index is None:
                unit_index[ingredient.unit] = len(buckets)
                buckets.append(
                    {"unit": ingredient.unit, "quantities": [ingredient.quantity]}
                )
            else:
                buckets[existing_index]["quantities"].append(ingredient.quantity)

    normalized: list[NormalizedIngredient] = []
    for name, buckets in buckets_by_name.items():
        status = IngredientStatus.HAVE if name in have_marks else IngredientStatus.NEED
        for bucket in buckets:
            quantities: list[float | None] = bucket["quantities"]
            non_none_quantities = [q for q in quantities if q is not None]
            merged_quantity = sum(non_none_quantities) if non_none_quantities else None
            normalized.append(
                NormalizedIngredient(
                    name=name,
                    quantity=merged_quantity,
                    unit=bucket["unit"],
                    status=status,
                )
            )

    logger.debug(
        "normalize_ingredients_node produced %d rows for session %s",
        len(normalized),
        state["session_id"],
    )
    return {"normalized_ingredients": normalized, "have_marks": have_marks}
