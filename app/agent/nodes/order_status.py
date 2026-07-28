"""The order-status branch.

`route_turn` (CUE-86) routes "where is my order" turns here. The real node -
the one that reads the user's live orders and answers in prose - is CUE-88's
work; this is the deterministic stand-in that keeps the branch honest until
then, and it is the only reason this module exists.

It is model-free on purpose: an empty node body would strand the turn with no
reply, and a model call with nothing to ground it on would invent a delivery
time. Saying plainly that chat cannot answer it yet, and pointing at the
screen that can (`GET /orders`, already shipped), is the truthful answer
available today. CUE-88 replaces the body; the wiring and the route stay.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage

from app.agent.state import AgentState

logger = logging.getLogger(__name__)

ORDER_STATUS_PENDING_MESSAGE = (
    "I can't check on an order from chat yet - open the Orders tab and "
    "you'll see where it's got to. Ask me for a dish any time and I'll get "
    "the basket ready."
)


def order_status_node(state: AgentState) -> dict[str, Any]:
    """Answer an order-status turn.

    Placeholder pending CUE-88, which replaces this body with the real
    lookup. It is wired into `build_graph` now because `route_turn` declares
    `order_status` as one of its four destinations, and a `Command` whose
    target does not exist fails at compile time - so the route and the node
    have to land together.

    Args:
        state: The current graph state. Read only for logging.

    Returns:
        A partial state update appending the reply to the transcript.
    """
    logger.info(
        "Order-status turn for session %s (CUE-88 pending)", state["session_id"]
    )
    return {"messages": [AIMessage(content=ORDER_STATUS_PENDING_MESSAGE)]}
