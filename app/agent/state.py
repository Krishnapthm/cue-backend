from __future__ import annotations

from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from app.agent.schemas import GeneratedRecipe


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
    """

    session_id: str
    user_id: int
    messages: Annotated[list[BaseMessage], add_messages]
    recipe: NotRequired[GeneratedRecipe | None]
    image_object_path: NotRequired[str | None]
