from __future__ import annotations

import logging
from typing import Any

from app.agent.schemas import IngredientStatus, NormalizedIngredient
from app.agent.state import AgentState

logger = logging.getLogger(__name__)


def normalize_ingredients_node(state: AgentState) -> dict[str, Any]:
    """Reshape `GeneratedRecipe.ingredients` into structured have/need rows.

    Deterministic, not model-driven: maps `state["recipe"].ingredients` (from
    either intake path - CUE-22 dish-name generation or CUE-23 photo parse,
    both produce a `GeneratedRecipe`) plus the have-list checkbox state
    already captured in `state["have_marks"]` into
    `list[NormalizedIngredient]`. Every source ingredient produces a row -
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

    Args:
        state: The current graph state. Must have `recipe` set to a
            `GeneratedRecipe`; `have_marks` is optional and defaults to an
            empty set when absent.

    Returns:
        A partial state update containing `normalized_ingredients`. This is
        a partial dict rather than a full `AgentState` (LangGraph's node
        convention, see `smoke_test_node` in `app/agent/graph.py`) - a
        partial dict cannot satisfy the total `AgentState` TypedDict under
        strict typing.

    Raises:
        ValueError: `state["recipe"]` is missing or `None`, so there is
            nothing to normalize.
    """
    recipe = state.get("recipe")
    if recipe is None:
        raise ValueError(
            "Cannot normalize ingredients: state has no recipe to normalize."
        )
    have_marks = state.get("have_marks") or set()

    # name -> ordered list of unit buckets; each bucket accumulates the
    # quantities of every source row sharing that (name, unit) pair. Buckets
    # with unit=None are never reused - each None-unit row gets its own
    # bucket - so they never merge with one another.
    buckets_by_name: dict[str, list[dict[str, Any]]] = {}
    # name -> unit -> index of that unit's bucket in buckets_by_name[name],
    # used to find the existing bucket to merge a repeated non-None unit into.
    unit_bucket_index: dict[str, dict[str, int]] = {}

    for ingredient in recipe.ingredients:
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
    return {"normalized_ingredients": normalized}
