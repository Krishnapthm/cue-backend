from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class RecipeIngredient(BaseModel):
    """A single ingredient line within a generated recipe."""

    name: str
    quantity: float | None = None
    unit: str | None = None


class GeneratedRecipe(BaseModel):
    """A structured, LLM-generated recipe for a given dish name.

    `method_summary` is deliberately brief prose rather than full step-by-step
    instructions - per R1.2 it exists as the hallucination backstop (forcing
    the model to reconcile ingredients against a coherent method) rather than
    as user-facing cooking instructions.
    """

    dish_name: str
    estimated_time_minutes: int
    ingredients: list[RecipeIngredient]
    method_summary: str


class ScopeVerdict(StrEnum):
    """Whether a user turn is something Cue is willing to act on."""

    IN_SCOPE = "in_scope"
    OUT_OF_SCOPE = "out_of_scope"


class GuardrailDecision(BaseModel):
    """The entry guardrail's classification of one user turn.

    `verdict` is a closed enum on purpose: a classifier that returns free
    text or an unrecognized label fails Pydantic validation and is handled
    as a malformed response, rather than being coerced into "in scope".

    `reason` is a short, model-controlled rationale kept for logs and traces
    only. It is never rendered to the user - concatenating attacker-
    influenced text into the reply would reintroduce the very injection
    vector this node exists to close.
    """

    verdict: ScopeVerdict
    reason: str


class IngredientStatus(StrEnum):
    """Whether a normalized ingredient is already owned by the user."""

    HAVE = "have"
    NEED = "need"


class NormalizedIngredient(BaseModel):
    """A single ingredient row after `normalize_ingredients_node`.

    This is the exact shape variant selection (R4.2) consumes - no downstream
    adapter reshapes it further. Only rows with `status == NEED` are handed
    to matching; `HAVE` rows are kept for display.
    """

    name: str
    quantity: float | None = None
    unit: str | None = None
    status: IngredientStatus
