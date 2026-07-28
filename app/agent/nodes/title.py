"""Best-effort, one-time naming of chat sessions after their resolved dish."""

from __future__ import annotations

import asyncio
import logging
import uuid

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from sqlalchemy import select, update

from app.agent.config import ModelRole
from app.agent.context import CueContext
from app.agent.providers import get_chat_model
from app.agent.state import AgentState
from app.database import session_factory
from app.models.chat import ChatSession

logger = logging.getLogger(__name__)
_title_tasks: set[asyncio.Task[None]] = set()

_SYSTEM_PROMPT = (
    "Write a concise chat title for a cooking session. Return only the dish "
    "name in 2-4 words, with no quotes and no trailing punctuation."
)


def _title_text(message: BaseMessage) -> str:
    """Normalize model output to the database title constraint."""
    content = message.content
    raw = content if isinstance(content, str) else str(content)
    title = " ".join(raw.strip().strip("'\"").split())
    title = title.rstrip(".!?;:")
    title = " ".join(title.split()[:4])
    return title[:200].rstrip()


async def generate_title(session_id: uuid.UUID, dish_name: str) -> None:
    """Generate and atomically persist a title without affecting a chat turn.

    The initial read avoids spending a model call for an already-titled
    session. The guarded update is still needed to make concurrent tasks
    across workers write exactly once.
    """
    try:
        async with session_factory() as session:
            existing = await session.scalar(
                select(ChatSession.title).where(ChatSession.id == session_id)
            )
        if existing is not None:
            return

        response = await get_chat_model(ModelRole.TITLE).ainvoke(
            [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=dish_name)]
        )
        title = _title_text(response)
        if not title:
            logger.warning(
                "Title model returned an empty title for session %s", session_id
            )
            return

        async with session_factory() as session:
            written_session_id = await session.scalar(
                update(ChatSession)
                .where(ChatSession.id == session_id, ChatSession.title.is_(None))
                .values(title=title)
                .returning(ChatSession.id)
            )
            await session.commit()
        if written_session_id is not None:
            logger.info("Named chat session %s", session_id)
    except Exception:
        # Titling is optional metadata. A provider or persistence failure must
        # leave the cooking turn entirely unaffected and keep title nullable.
        logger.exception("Could not generate title for chat session %s", session_id)


def _schedule_title_generation(session_id: uuid.UUID, dish_name: str) -> None:
    """Start title generation without holding up the graph's critical path."""
    task = asyncio.create_task(generate_title(session_id, dish_name))
    _title_tasks.add(task)
    task.add_done_callback(_title_tasks.discard)


async def schedule_title_node(
    state: AgentState, runtime: Runtime[CueContext]
) -> dict[str, bool]:
    """Schedule one non-blocking title attempt after resolving a recipe.

    Args:
        state: The checkpointed graph state.
        runtime: The invocation runtime carrying the chat session identity.

    Returns:
        A partial update recording that this session has already attempted
        title generation, or an empty update when it has.
    """
    if state.get("title_attempted"):
        return {}
    recipe = state.get("recipe")
    if recipe is None:
        return {"title_attempted": True}
    try:
        _schedule_title_generation(runtime.context.chat_session_id, recipe.dish_name)
    except Exception:
        logger.exception(
            "Could not schedule title generation for chat session %s",
            runtime.context.chat_session_id,
        )
    return {"title_attempted": True}
