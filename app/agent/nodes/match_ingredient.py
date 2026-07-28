"""The ingredient fan-out: one parallel worker per NEED ingredient (CUE-91).

Once the user has answered the checklist, every ingredient they did *not* mark
as owned has to become a real, purchasable Swiggy variant. Those are
independent network calls with no ordering between them, so they fan out with
`Send` - which is both the fast implementation and the one the chat design
draws, a row at a time.

Every piece of reasoning here already existed and had no caller:
`instamart.search_products` (CUE-11), `cart.select_variant` (CUE-15) and
`matching.propose_substitute` (CUE-17). This module is the wiring, not new
matching logic, and deliberately adds no ranking rules of its own - two rankers
that could disagree about what an ingredient resolves to is exactly the drift
this codebase is most exposed to.

**`AgentState.matches` must keep its `operator.add` reducer** (CUE-85). The
workers write it in parallel, and without the reducer the last one to finish
silently overwrites every other result. That failure produces a plausible
one-item cart rather than an error, which is the dangerous kind.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END
from langgraph.runtime import Runtime
from langgraph.types import Send

from app.agent.context import CueContext
from app.agent.schemas import IngredientStatus, MatchResult, NormalizedIngredient
from app.agent.state import AgentState
from app.cart import service as cart_service
from app.cart.schemas import Ingredient, MatchStatus, SelectedVariant
from app.instamart import service as instamart_service
from app.instamart.exceptions import InstamartDomainError
from app.matching import substitution as matching_service

logger = logging.getLogger(__name__)

#: The node name the fan-out sends to. Defined here rather than imported from
#: `app.agent.graph` because the graph imports this module - the name has to
#: live on the side that has no dependency on the other.
MATCH_INGREDIENT = "match_ingredient"


class MatchTask(TypedDict):
    """The payload one `Send` carries into one worker.

    A worker is handed a single ingredient rather than the whole state: it
    needs nothing else, and keeping the payload minimal keeps what LangGraph
    checkpoints per parallel branch small.
    """

    ingredient: NormalizedIngredient


def needed_ingredients(state: AgentState) -> list[NormalizedIngredient]:
    """Return the ingredients this turn has to buy.

    Keyed on `have_marks` rather than on each row's `status`, and the
    distinction matters: `normalized_ingredients` was built *before* the
    interrupt, with statuses derived from the pantry-seeded marks, while
    `have_marks` carries whatever the user actually answered on the checklist.
    After a resume the two disagree wherever the user changed a tick, and the
    user's answer is the one that authorizes spending their money.

    Falls back to the row's own status only when there are no marks at all -
    a state literal built before the checklist existed, which is what the unit
    tests of the downstream nodes use.

    Args:
        state: The graph state, after `confirm_checklist` has resumed.

    Returns:
        The rows to resolve against Instamart, in recipe order.
    """
    rows = state.get("normalized_ingredients") or []
    marks = state.get("have_marks")
    if marks is None:
        return [row for row in rows if row.status is IngredientStatus.NEED]
    return [row for row in rows if row.name not in marks]


def _as_ingredient(row: NormalizedIngredient) -> Ingredient:
    """Project a checklist row onto the shape `select_variant` consumes.

    `preferred_brand` is left unset: the go-to brand signal (R4.3) is sourced
    from order history, which is not wired into the graph, and inventing one
    here would quietly change what gets bought.
    """
    return Ingredient(
        name=row.name,
        quantity=Decimal(str(row.quantity)) if row.quantity is not None else None,
        unit=row.unit,
    )


def _unavailable(row: NormalizedIngredient, reason: str) -> MatchResult:
    """Build the row for an ingredient nothing purchasable could be found for.

    An unavailable ingredient is represented, never dropped. A cart that
    silently omits an item looks complete and is not - the user finds out at
    the stove, which is the one failure this whole path exists to avoid.
    """
    return MatchResult(
        ingredient_name=row.name,
        status=MatchStatus.UNAVAILABLE,
        substitution_reason=reason[:1000],
    )


def _from_selected(row: NormalizedIngredient, selected: SelectedVariant) -> MatchResult:
    """Project `select_variant`'s choice onto the graph's match row.

    `substitution_reason` is carried only for a substituted row, per
    `MatchResult`: it answers "why is this not what you asked for?", and a
    matched row is not a swap and has nothing to explain.
    """
    substituted = selected.match_status is MatchStatus.SUBSTITUTED
    return MatchResult(
        ingredient_name=row.name,
        status=selected.match_status,
        spin_id=selected.spin_id,
        product_name=selected.product_name,
        pack_size=selected.pack_size,
        unit_price=selected.unit_price,
        quantity=selected.quantity,
        substitution_reason=selected.selection_reason[:1000] if substituted else None,
    )


async def _resolve(context: CueContext, row: NormalizedIngredient) -> MatchResult:
    """Resolve one ingredient to a variant, a substitute, or nothing.

    `search_products` -> `select_variant`, and on "nothing in stock" a second
    pass through `propose_substitute`.

    That second pass is the ticket's design and is kept, but it is worth being
    honest about its reach: `propose_substitute` re-searches the *same* term,
    and its purchasable filter (in stock **and** priced) is strictly narrower
    than `select_variant`'s (in stock), so on today's code it can only confirm
    the same answer at the cost of one more Swiggy call. It earns its place as
    the backstop for the case the filters diverge - and it is the only path
    that produces a substitution *reason* rather than a bare swap.
    """
    products = await instamart_service.search_products(
        context.session,
        context.user_id,
        address_id=context.address_id,
        query=row.name,
    )
    selected = cart_service.select_variant(_as_ingredient(row), products)
    if selected.match_status is not MatchStatus.UNAVAILABLE:
        return _from_selected(row, selected)

    substitute = await matching_service.propose_substitute(
        context.session,
        context.user_id,
        context.address_id,
        row.name,
        selected.pack_size,
        selected.quantity or 1,
    )
    if substitute is None:
        return _unavailable(row, selected.selection_reason)

    return MatchResult(
        ingredient_name=row.name,
        status=MatchStatus.SUBSTITUTED,
        spin_id=substitute.spin_id,
        product_name=substitute.product_name,
        pack_size=substitute.pack_size,
        unit_price=substitute.unit_price,
        quantity=substitute.quantity,
        substitution_reason=substitute.reason,
    )


def fan_out(state: AgentState) -> list[Send] | str:
    """Fan one `Send` out per ingredient the user still needs.

    Used as the conditional edge off `confirm_checklist`, so the workers all
    start in the same super-step and run concurrently. Nothing here may
    interrupt: this runs *after* the graph's only pause, and the fan-out is
    where the user's consent is spent, not where it is asked for.

    Args:
        state: The graph state, after the checklist has been answered.

    Returns:
        One `Send` per NEED ingredient, or `END` when there are none - a user
        who already owns everything has nothing to buy, and an empty `Send`
        list would strand the turn with no next node at all.
    """
    rows = needed_ingredients(state)
    if not rows:
        logger.info(
            "Nothing to buy for session %s: every ingredient is already owned.",
            state["session_id"],
        )
        return END
    return [Send(MATCH_INGREDIENT, MatchTask(ingredient=row)) for row in rows]


async def match_ingredient(
    state: MatchTask, *, runtime: Runtime[CueContext]
) -> dict[str, list[MatchResult]]:
    """Resolve one ingredient against Instamart and stream the result.

    One worker, one ingredient, one row appended to `matches`. Workers finish
    out of order and that is fine: the event carries the ingredient name as a
    stable key, so the client fills its row in place rather than appending.
    Serializing them to make the stream tidy would trade the feature for the
    ordering.

    All three outcomes are user-visible. Silently swapping a brand or a pack
    size is the fastest way to lose a user's trust in the cart, and an
    ingredient that quietly vanished is worse.

    Failure isolation is deliberate and asymmetric:

    * `InstamartDomainError` - Swiggy executed the search and said no (store
      closed, term rejected). That is *this* ingredient's permanent failure,
      so it becomes an `unavailable` row and every other row still resolves.
    * `InstamartAuthError` and `InstamartTransportError` propagate. The token
      being dead, or Swiggy being unreachable after `NETWORK_RETRY`'s
      attempts, is not a fact about one ingredient - every remaining worker is
      failing the same way. Ending the turn once on the reconnect/retry path
      beats N identical failures, and beats the alternative: composing a cart
      that is missing real items because Swiggy was down, which the user would
      only discover at the stove.

    Args:
        state: The `Send` payload carrying this worker's ingredient. Named
            `state` because that is what it is to LangGraph - a `Send` payload
            becomes the receiving node's whole state - and because the node
            protocol matches on the parameter's name.
        runtime: Supplies `CueContext` - the request's session, user and the
            address search is scoped to.

    Returns:
        A single-element `matches` update. The `operator.add` reducer on that
        field is what turns N of these into N rows; see the module docstring.
    """
    row = state["ingredient"]
    try:
        result = await _resolve(runtime.context, row)
    except InstamartDomainError as exc:
        logger.info(
            "Swiggy rejected the search for %r; recording it as unavailable (%s).",
            row.name,
            exc.detail,
        )
        result = _unavailable(row, exc.detail)

    # Emitted before the return so the row reaches the client the moment it is
    # known, rather than when the whole super-step's state update is applied.
    get_stream_writer()(result.model_dump(mode="json"))
    return {"matches": [result]}
