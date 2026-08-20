"""The refusal path.

The classifier that used to live here is now `app/agent/nodes/route.py`: a
four-path graph needs its entry node to answer a wider question than "in scope
or not". What stays is the branch that turns a refused turn into a reply -
deliberately model-free, for the reasons in `refuse_node`.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage

from app.agent.state import AgentState

logger = logging.getLogger(__name__)

#: One line, because a refusal is a boundary and not a briefing. The long
#: version recited what Cue does and how to ask for it, which reads as a
#: lecture even when the turn deserved a no - and pleasantries no longer reach
#: this node at all, so the message no longer has to carry the friendly case
#: too. See `app/agent/nodes/small_talk.py`.
REFUSAL_MESSAGE = "That one's outside what I do - but I'm all yours for cooking."


def refuse_node(state: AgentState) -> dict[str, Any]:
    """Append a fixed, non-model refusal message.

    Deterministic and template-based on purpose: the refusal path must not
    make a second model call with attacker-influenced text in the prompt.
    For the same reason the reply never quotes the user's message back, and
    never renders `GuardrailDecision.reason` - both are attacker-controlled
    strings, and echoing either would reopen the hole this path exists to
    close.

    `state["recipe"]` is deliberately left untouched: a refused turn must
    not disturb whatever the session already had.

    Args:
        state: The current graph state. Read only for logging.

    Returns:
        A partial state update appending the refusal to the transcript.
    """
    logger.info("Refusing out-of-scope turn for session %s", state["session_id"])
    return {"messages": [AIMessage(content=REFUSAL_MESSAGE)]}
