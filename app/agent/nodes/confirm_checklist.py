"""The graph's only interrupt: which of these do you already have? (CUE-90)

This is the one point where the graph stops and asks the user something, and
their answer decides what gets bought.

It is also the only interrupt in the graph. Checkout left the graph entirely and
is a plain REST call from the cart screen (CUE-80), which is worth stating
plainly: with no pause in front of a non-idempotent mutation, the double-charge
failure mode is structurally absent rather than carefully defended against.

**`interrupt()` is the first statement in the node body, and must stay that
way.** On resume LangGraph restarts the node from the top, so every statement
before `interrupt()` runs again - once per resume. This node interrupts
immediately, so re-execution costs nothing and can corrupt nothing. Any future
"just one quick thing before we ask" - a log line, a DB write, a message append -
is a duplicate-write bug waiting to happen.

The payload is built by `checklist_payload`, called in the `interrupt()`
argument. That keeps it one statement, and the builder is pure: re-running it on
resume produces the same payload and touches nothing.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.types import interrupt

from app.agent.schemas import ChecklistDecision, ChecklistInterrupt, ChecklistItem
from app.agent.state import AgentState

logger = logging.getLogger(__name__)


def checklist_payload(state: AgentState) -> dict[str, Any]:
    """Build the interrupt payload: the checklist the user is about to answer.

    Pure, and deliberately so - see the module docstring on re-execution.

    Args:
        state: The current graph state, after `normalize_ingredients`.

    Returns:
        The JSON-serializable payload. `ui: "checklist"` discriminates it: the
        design renders this interrupt inline in the chat transcript and future
        interrupts may not, so the client routes on an explicit field rather
        than inferring intent from the payload's shape.
    """
    rows = state.get("normalized_ingredients") or []
    return ChecklistInterrupt(
        items=[ChecklistItem.from_normalized(row) for row in rows]
    ).model_dump(mode="json")


def confirm_checklist(state: AgentState) -> dict[str, Any]:
    """Pause and ask the user which ingredients they already have.

    The resume value is also the user's consent to mutate their Swiggy cart:
    everything downstream of this node runs only because they pressed the
    button. Swiggy's `update_cart` is a full replace on a server-side cart the
    user may also be editing in the Swiggy app, so confirm-before-mutate is not
    ceremony.

    Args:
        state: The current graph state, after `normalize_ingredients`.

    Returns:
        A partial state update carrying the user's marks, which replace the
        pantry-seeded ones `normalize_ingredients` wrote.

    Raises:
        ValidationError: The resume value was not a `{"have": [...]}` payload.
            Raised rather than coerced: a resume we cannot read is not consent,
            and defaulting to "none of them" would silently buy the user
            everything they already own.
    """
    marks = interrupt(checklist_payload(state))

    decision = ChecklistDecision.model_validate(marks)
    logger.info(
        "Checklist confirmed for session %s with %d have-marks",
        state["session_id"],
        len(decision.have),
    )
    return {"have_marks": set(decision.have)}
