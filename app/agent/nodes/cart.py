"""Push the plan to Swiggy, and close the turn (CUE-92).

This is where the agent's work becomes a real cart, and it is the last thing
the graph does.

`compose_cart` marshals; it does not decide. `cart.service.compose_cart`
(CUE-16) already supersedes the previous live plan, writes `CartPlan` +
`CartPlanItem`, commits, and pushes `update_cart` followed by a `get_cart`
read-back. None of that is reimplemented here - the node's whole job is
turning `matches` into `SelectedVariant` rows and handing them over.

`report_cart` renders the outcome and nothing else. In particular it does
*not* persist: the `CART_READY` message is written by `chat.service`, which
owns every transcript write, exactly as it owns the `CHECKLIST` write on the
pause. A node reaching into `chat.service` would also close an import cycle,
since `chat.service` imports `agent.graph` which imports this module.

**The graph does not check out.** "Checkout on Instamart" is a REST call from
the cart screen against CUE-80's endpoints, and `place_order` (CUE-19) already
handles the hard part - it writes `status="unknown"` on transport failure and
reconciles against order history rather than retrying a non-idempotent call.
There is deliberately no checkout node here: that separation is what makes a
double charge impossible rather than merely unlikely.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

from langgraph.runtime import Runtime

from app.agent.context import CueContext
from app.agent.schemas import CartReport, CartReportItem, MatchResult
from app.agent.state import AgentState
from app.cart import service as cart_service
from app.cart.schemas import ComposeCartResult, MatchStatus, SelectedVariant
from app.instamart.schemas import CartLineItem

logger = logging.getLogger(__name__)

#: The node names, defined beside their nodes rather than in `app.agent.graph`:
#: the graph imports this module, so the names have to live on the side that
#: has no dependency on the other. `fan_out` routes to `COMPOSE_CART`.
COMPOSE_CART = "compose_cart"
REPORT_CART = "report_cart"

#: Used when a match somehow carries no reason of its own. `cart_plan_item`
#: allows a null reason, but a plan row that cannot say why it exists is a
#: worse audit trail than one that admits it does not know.
UNRECORDED_SELECTION_REASON = "Selected without a recorded reason."


def _as_selected_variant(
    match: MatchResult, quantity: Decimal | None, unit: str | None
) -> SelectedVariant:
    """Project one match row onto the shape `compose_cart` persists.

    The needed quantity and unit come from the checklist row rather than from
    the match: `MatchResult` records what is being *bought* (pack size, pack
    count), while `cart_plan_item.ingredient_qty/unit` record what the recipe
    *asked for*. They are different numbers and the plan keeps both.
    """
    return SelectedVariant(
        ingredient_name=match.ingredient_name,
        ingredient_qty=quantity,
        ingredient_unit=unit,
        match_status=match.status,
        spin_id=match.spin_id,
        product_name=match.product_name,
        pack_size=match.pack_size,
        unit_price=match.unit_price,
        image_url=match.image_url,
        rating=match.rating,
        quantity=match.quantity,
        selection_reason=match.selection_reason or UNRECORDED_SELECTION_REASON,
    )


async def compose_cart_node(
    state: AgentState, runtime: Runtime[CueContext]
) -> dict[str, Any]:
    """Turn this turn's matches into a `CartPlan` and a real Swiggy cart.

    `address_id` comes off `CueContext`, not from state and not from a lookup
    here. The graph does not resolve addresses; `run_turn` guarantees one is
    present before it invokes at all, because Swiggy binds a cart to an address
    and a turn without one could never finish.

    Not retried at the node level, unlike the off-box read paths. This node
    writes: a retry would supersede its own just-written plan and insert
    another, so a transient failure here is better surfaced as a retryable turn
    than silently turned into duplicate plan history.

    Args:
        state: The graph state, after the ingredient fan-out has finished.
        runtime: Supplies `CueContext` - the request's session, user, chat
            session and delivery address.

    Returns:
        The plan id and the compose verdict `report_cart` renders from.
    """
    context = runtime.context
    rows = {row.name: row for row in state.get("normalized_ingredients") or []}
    matches = state.get("matches") or []

    selections = []
    for match in matches:
        row = rows.get(match.ingredient_name)
        quantity = (
            Decimal(str(row.quantity)) if row and row.quantity is not None else None
        )
        selections.append(
            _as_selected_variant(match, quantity, row.unit if row else None)
        )

    result = await cart_service.compose_cart(
        context.session,
        context.user_id,
        context.chat_session_id,
        context.address_id,
        selections,
    )
    logger.info(
        "Composed plan %s for session %s: %d item(s), subtotal %s, below_minimum=%s",
        result.plan_id,
        context.chat_session_id,
        len(selections),
        result.subtotal,
        result.below_minimum,
    )
    return {"cart_plan_id": result.plan_id, "compose_result": result}


def _money(value: Decimal) -> str:
    """Render a rupee amount the way the card shows it."""
    return f"₹{value:.2f}"


def _report_item(match: MatchResult, line: CartLineItem | None) -> CartReportItem:
    """Build one card row, preferring Swiggy's cart line over our snapshot."""
    return CartReportItem(
        ingredient_name=match.ingredient_name,
        status=match.status,
        in_cart=line is not None,
        product_name=match.product_name,
        pack_size=match.pack_size,
        quantity=line.quantity if line is not None else match.quantity,
        unit_price=match.unit_price,
        image_url=match.image_url,
        rating=match.rating,
        line_total=line.price if line is not None else None,
        substitution_reason=match.substitution_reason,
    )


def _summary(result: ComposeCartResult, items: list[CartReportItem]) -> str:
    """Write the one-line headline the chat bubble leads with.

    Deterministic prose, not a model call: this sentence states what was spent
    and what is missing, and a model has nothing to add to that but risk.
    """
    if not items:
        return (
            "You already have everything this recipe needs - there was nothing to "
            "add to your cart."
        )
    if result.below_minimum:
        return (
            f"Your cart comes to {_money(result.subtotal)}, which is "
            f"{_money(result.shortfall)} under Instamart's "
            f"{_money(result.minimum_order_value)} minimum. Add a little more on "
            "the cart screen and it's ready to check out."
        )

    in_cart = [item for item in items if item.in_cart]
    total = result.cart.total if result.cart is not None else None
    headline = f"Cart ready: {len(in_cart)} item{'' if len(in_cart) == 1 else 's'}"
    if total is not None:
        headline += f", {_money(total)}"

    notes = []
    substituted = sum(1 for item in in_cart if item.status is MatchStatus.SUBSTITUTED)
    if substituted:
        notes.append(f"{substituted} substituted")
    missing = [item for item in items if not item.in_cart]
    if missing:
        notes.append(f"{len(missing)} unavailable")
    if notes:
        headline += f". {' and '.join(notes)} - tap to see why"
    return f"{headline}."


def report_cart_node(state: AgentState) -> dict[str, Any]:
    """Render the turn's closing card from the cart Swiggy actually holds.

    **Never renders from what it believes it just sent.** Swiggy's cart is
    server-side state the user may also be editing in the Swiggy app, and
    Swiggy can answer `success: true` while quietly omitting a line it could
    not honour. `compose_cart` already re-reads with `get_cart`; this renders
    that response, so a dropped line shows up as `in_cart=False` rather than as
    a row the user believes is bought.

    Below the minimum, the shortfall is reported and nothing is chased. There
    is no `suggest_addons` node and no retry loop - the graph stays acyclic,
    and topping up an order is the user's call.

    Args:
        state: The graph state, after `compose_cart`.

    Returns:
        The `cart_report` the turn ends on. `chat.service` persists it as a
        `CART_READY` message; see the module docstring on why not here.

    Raises:
        RuntimeError: `compose_cart` did not run before this node, which means
            the graph changed shape without this call site being updated.
    """
    result = state.get("compose_result")
    if result is None:
        raise RuntimeError(
            "report_cart ran with no compose_cart result: the graph's edges "
            "changed without this node being updated."
        )

    lines: dict[str, CartLineItem] = {
        line.spin_id: line for line in (result.cart.items if result.cart else [])
    }
    items = [
        _report_item(match, lines.get(match.spin_id) if match.spin_id else None)
        for match in state.get("matches") or []
    ]

    report = CartReport(
        plan_id=result.plan_id,
        summary=_summary(result, items),
        below_minimum=result.below_minimum,
        subtotal=result.subtotal,
        minimum_order_value=result.minimum_order_value,
        shortfall=result.shortfall,
        cart_total=result.cart.total if result.cart is not None else None,
        items=items,
    )
    return {"cart_report": report}
