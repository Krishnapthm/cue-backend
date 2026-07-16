from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Shared state threaded through every graph node.

    `session_id` is `str(chat_session.id)` and doubles as the LangGraph
    `thread_id` the checkpointer keys on (see `app/models/chat.py`).

    `messages` carries the conversation. It uses the `add_messages` reducer
    rather than a plain list so node returns *append to* (and upsert by id)
    the transcript instead of overwriting it - the correct behaviour once the
    checkpointer replays state across turns and later nodes emit messages.
    """

    session_id: str
    user_id: int
    messages: Annotated[list[BaseMessage], add_messages]
