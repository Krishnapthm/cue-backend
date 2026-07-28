from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from app.cart.schemas import MatchStatus


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


class TurnIntent(StrEnum):
    """What the user's turn is asking the graph to do (CUE-86).

    A closed enum for the same reason `ScopeVerdict` is one: an entry
    classifier that returns an unrecognized label must fail validation and be
    handled as malformed, never coerced into a path that spends a model call
    or touches the user's cart.
    """

    OUT_OF_SCOPE = "out_of_scope"
    RECIPE = "recipe"
    PHOTO = "photo"
    ORDER_STATUS = "order_status"


class TurnClassification(BaseModel):
    """The entry router's read of one user turn (CUE-86).

    The structured-output schema `route_turn` asks the router model for. It is
    restated as a `GuardrailDecision` for state and traces, which is why that
    model is unchanged by the four-path refactor.

    `reason` is a short, model-controlled rationale kept for logs and traces
    only, and carries the same rule as `GuardrailDecision.reason`: it is never
    rendered to the user, because concatenating attacker-influenced text into
    the reply would reintroduce the very injection vector this node exists to
    close.
    """

    intent: TurnIntent
    reason: str


class MatchResult(BaseModel):
    """One ingredient resolved against Instamart, as the graph carries it.

    Written by the parallel `Send` workers of the ingredient fan-out, so
    `AgentState.matches` reduces with `operator.add` - see the state docstring.
    Everything here is JSON-serializable because it is checkpointed.

    `status` reuses `cart.schemas.MatchStatus` rather than redeclaring the
    three values: that enum mirrors `cart_plan_item.match_status`'s CHECK
    constraint, and a second copy would be free to drift away from the column
    these rows are ultimately persisted into.

    `substitution_reason` is why a swap was made ("Amul 500g out of stock;
    nearest pack is Nandini 450g"), so the checklist can say *why* something
    changed rather than silently showing a different product. It is
    deterministic, service-generated text - unlike `GuardrailDecision.reason`
    it is safe to render. It is `None` on a plain `MATCHED` row.
    """

    ingredient_name: str
    status: MatchStatus
    spin_id: str | None = None
    product_name: str | None = None
    pack_size: str | None = None
    unit_price: Decimal | None = None
    quantity: int | None = None
    substitution_reason: str | None = Field(default=None, max_length=1000)


class TurnFailureKind(StrEnum):
    """Why a turn could not be completed, at the granularity the UI acts on."""

    # The Swiggy link expired or was revoked: the only recovery is a fresh
    # OAuth authorize (see `app.instamart.exceptions.InstamartAuthError`).
    PROVIDER_AUTH = "provider_auth"
    # Swiggy was unreachable or errored; retrying the turn is reasonable.
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    # Swiggy executed the request and said no (out of stock, store closed).
    PROVIDER_REJECTED = "provider_rejected"
    # Anything else. The client offers a retry and nothing more specific.
    INTERNAL = "internal"


class TurnFailure(BaseModel):
    """A turn that ended without its intended output, in renderable form.

    Recorded in state (rather than raised out of the graph) for failures the
    user is expected to *act* on - reconnecting Swiggy, above all - because a
    turn that streamed half a checklist before failing has already sent its
    HTTP status code. `message` is written by us, never by a model and never
    from a raw upstream error, so it is safe to render.
    """

    kind: TurnFailureKind
    message: str


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


class ChecklistItem(BaseModel):
    """One row of the checklist the `confirm_checklist` interrupt renders.

    A projection of `NormalizedIngredient` rather than the model itself: this
    crosses to the client, so it carries `have` as the boolean a checkbox binds
    to instead of the internal `IngredientStatus` enum. `have` arrives already
    ticked for anything the pantry seeded (CUE-89).
    """

    name: str
    quantity: float | None = None
    unit: str | None = None
    have: bool

    @classmethod
    def from_normalized(cls, row: NormalizedIngredient) -> ChecklistItem:
        """Project one normalized row onto its checklist row."""
        return cls(
            name=row.name,
            quantity=row.quantity,
            unit=row.unit,
            have=row.status is IngredientStatus.HAVE,
        )


class ChecklistInterrupt(BaseModel):
    """The payload `confirm_checklist` interrupts with (CUE-90).

    `ui` is a discriminator, not decoration: the design renders this interrupt
    inline in the chat transcript and a future interrupt may not, so the client
    routes on an explicit field rather than inferring from the payload's shape.
    """

    ui: Literal["checklist"] = "checklist"
    items: list[ChecklistItem]


class ChecklistDecision(BaseModel):
    """The user's answer to the checklist, i.e. the interrupt's resume value.

    `have` is the ingredient names they confirmed owning; everything else on the
    checklist is bought. Validated rather than read as a raw dict because it is
    the consent that authorizes mutating the user's Swiggy cart.
    """

    have: list[str]
