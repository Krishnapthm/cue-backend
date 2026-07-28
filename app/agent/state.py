from __future__ import annotations

import operator
from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from app.agent.schemas import (
    CartReport,
    GeneratedRecipe,
    GuardrailDecision,
    MatchResult,
    NormalizedIngredient,
    TurnFailure,
    TurnIntent,
)
from app.cart.schemas import ComposeCartResult


class AgentState(TypedDict):
    """Shared state threaded through every graph node.

    `session_id` is `str(chat_session.id)` and doubles as the LangGraph
    `thread_id` the checkpointer keys on (see `app/models/chat.py`).

    `messages` carries the conversation. It uses the `add_messages` reducer
    rather than a plain list so node returns *append to* (and upsert by id)
    the transcript instead of overwriting it - the correct behaviour once the
    checkpointer replays state across turns and later nodes emit messages.

    `recipe` is `NotRequired` (rather than `GeneratedRecipe | None` on a total
    TypedDict) so state literals built before recipe generation existed - and
    partial node updates that never touch it - stay valid without every
    caller having to pass `recipe=None` explicitly.

    `image_object_path` is the Supabase Storage object path of an uploaded
    recipe photo, sourced from `ChatMessage.payload` (kind='image', see
    `app/models/chat.py`). It is `NotRequired` for the same reason as
    `recipe`: most turns never carry an image, and the payload -> state
    extraction at the app/graph boundary is wired in a later issue (this
    field only defines the contract `parse_recipe_photo_node` reads from).

    `have_marks` is the set of ingredient names marked as already-owned for the
    current recipe. It has two sources and a strict precedence between them:
    the user's own answer (a `confirm_checklist` resume, or an explicit client
    submission) always wins, and only when there is none does
    `normalize_ingredients_node` seed it from the user's in-stock `PantryItem`
    rows so the obvious staples arrive pre-ticked. Names are spelled as the
    *recipe* spells them, not as the pantry does, since that is what every
    reader matches against. It is `NotRequired` for the same reason as
    `recipe`: most state literals never touch it.

    The recipe-producing nodes clear it, because it outlives a turn in the
    checkpoint while the checklist it describes does not - see
    `generate_recipe_node`'s `Returns` for what goes wrong otherwise.

    `normalized_ingredients` is `normalize_ingredients_node`'s output: the
    same `GeneratedRecipe.ingredients` reshaped into `(name, quantity/unit,
    have/need)` rows. It is `NotRequired` because it only exists once that
    node has run.

    `guardrail` is the entry guardrail's verdict for the current turn, kept
    for logs and traces rather than for rendering - see `GuardrailDecision`
    on why its `reason` never reaches the user. `NotRequired` for the same
    reason as `recipe`: state literals built before the guardrail existed
    stay valid without passing `guardrail=None`.

    `turn_intent` is `route_turn`'s verdict for the current turn: which of the
    four entry paths it took. `guardrail` keeps the in-scope/out-of-scope
    framing for logs and traces; this is the branch that was actually run.

    `matches` is the ingredient fan-out's output, one row per ingredient. It
    **must** carry the `operator.add` reducer: the rows are written by
    parallel `Send` workers, and without a reducer the last worker to finish
    silently overwrites every other worker's result - a failure that produces
    a plausible-looking one-item checklist rather than an error. It is
    `NotRequired` *inside* the `Annotated` so the reducer stays the outermost,
    visible thing while state literals written before it existed stay valid;
    the reducer supplies `[]` at runtime either way.

    `cart_plan_id` and `compose_result` are the cart composition's outcome:
    the `cart_plan` row it recorded, and the subtotal/minimum-order verdict
    the report is rendered from.

    `cart_report` is `report_cart`'s rendering of that outcome - the card the
    turn ends on. The node builds it; `chat.service` is what persists it as a
    `CART_READY` message, for the same reason the checklist pause works that
    way: every transcript write belongs to one owner, and a node that reached
    into `chat.service` would close an import cycle through `agent.graph`.

    `failure` records a turn that ended without its intended output in a form
    the client can render and act on (reconnect Swiggy, retry). Failures that
    are *our* bugs still raise and surface as 5xx; this field is for the ones
    the user is expected to resolve.

    Nothing here may be non-JSON-serializable: this TypedDict is checkpointed
    to Postgres in full. Request-scoped handles - the `AsyncSession` above all
    - travel on `CueContext` instead (see `app/agent/context.py`).
    """

    session_id: str
    user_id: int
    messages: Annotated[list[BaseMessage], add_messages]
    recipe: NotRequired[GeneratedRecipe | None]
    guardrail: NotRequired[GuardrailDecision | None]
    image_object_path: NotRequired[str | None]
    have_marks: NotRequired[set[str]]
    normalized_ingredients: NotRequired[list[NormalizedIngredient]]
    turn_intent: NotRequired[TurnIntent]
    matches: Annotated[NotRequired[list[MatchResult]], operator.add]
    cart_plan_id: NotRequired[int | None]
    compose_result: NotRequired[ComposeCartResult | None]
    cart_report: NotRequired[CartReport | None]
    failure: NotRequired[TurnFailure | None]
