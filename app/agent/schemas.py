from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from app.cart.schemas import MatchStatus
from app.instamart.schemas import ProductRating


class RecipeIngredient(BaseModel):
    """A single ingredient line within a generated recipe."""

    name: str
    quantity: float | None = None
    unit: str | None = None


class ScratchComponent(BaseModel):
    """One component that can meaningfully be made or bought ready-made."""

    name: str = Field(min_length=1)
    ready_made_name: str = Field(min_length=1)
    constituent_names: list[str] = Field(min_length=2)


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
    scratch_components: list[ScratchComponent] = Field(default_factory=list)


class ScratchChoice(StrEnum):
    """How the user wants to source a recipe component."""

    READY_MADE = "ready_made"
    FROM_SCRATCH = "from_scratch"


class ScratchChoiceOption(BaseModel):
    """One selectable option on a scratch-choice card."""

    id: ScratchChoice
    label: str


class ScratchChoiceInterrupt(BaseModel):
    """The pre-checklist choice card for one verified recipe component."""

    ui: Literal["scratch_choice"] = "scratch_choice"
    dish_name: str
    component_name: str
    ready_made_name: str
    options: list[ScratchChoiceOption]


class ScratchChoiceDecision(BaseModel):
    """The user's source choice for a verified recipe component."""

    choice: ScratchChoice


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

    `selection_reason` is the *full* reason the selection was made, including
    the pack-size arithmetic CUE-15 works out ("2 pack(s) leaves 100 g over the
    amount needed"). It is present on every row, and it is what
    `compose_cart` persists onto `cart_plan_item.selection_reason` - the audit
    trail for why the cart holds what it holds. `substitution_reason` is the
    subset of it that is worth putting in front of the user, which is why the
    two are separate rather than one field doing both jobs. Capped at 1000 to
    match that column's CHECK constraint, so it cannot violate it by
    construction.
    """

    ingredient_name: str
    status: MatchStatus
    spin_id: str | None = None
    product_name: str | None = None
    pack_size: str | None = None
    unit_price: Decimal | None = None
    image_url: str | None = None
    rating: ProductRating | None = None
    quantity: int | None = None
    substitution_reason: str | None = Field(default=None, max_length=1000)
    selection_reason: str | None = Field(default=None, max_length=1000)


class CartReportItem(BaseModel):
    """One line of the cart-ready card (CUE-92).

    `in_cart` is read back off Swiggy's own `get_cart`, not inferred from what
    we asked for. Swiggy does not always fail a write it cannot fully honour -
    it can answer `success: true` and quietly omit an out-of-stock line - so a
    line we sent that is not in the cart Swiggy returned is a real outcome the
    user has to be shown, not a rendering detail.

    `quantity` and `line_total` come from the cart line when there is one, and
    fall back to the compose-time snapshot only for rows that never made it
    into the cart. `unit_price` is always the snapshot: Swiggy reports a line
    total, and dividing it back out would invent precision we do not have.
    """

    ingredient_name: str
    status: MatchStatus
    in_cart: bool
    product_name: str | None = None
    pack_size: str | None = None
    quantity: int | None = None
    unit_price: Decimal | None = None
    image_url: str | None = None
    rating: ProductRating | None = None
    line_total: Decimal | None = None
    substitution_reason: str | None = None


class CartReport(BaseModel):
    """The terminal message of a cart turn: what is in the cart, and what is not.

    Persisted as the payload of a `MessageKind.CART_READY` message, so it has
    to stay JSON-serializable and self-contained - the transcript renders it
    again on a cold start, long after the turn's state is gone.

    `below_minimum` is reported, never chased. There is no `suggest_addons`
    node and no retry loop: the graph stays acyclic, and topping up an order is
    the user's call to make on the cart screen, not the agent's to make for
    them.
    """

    plan_id: int
    summary: str
    below_minimum: bool
    subtotal: Decimal
    minimum_order_value: Decimal
    shortfall: Decimal
    # Swiggy's own total for the cart it read back, when it gave one. `None`
    # below the minimum (nothing was pushed) or when Swiggy omitted it.
    cart_total: Decimal | None = None
    items: list[CartReportItem]


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
