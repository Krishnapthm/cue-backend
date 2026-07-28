from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agent.context import CueContext
from app.agent.nodes.guardrail import guardrail_node, refuse_node
from app.agent.nodes.recipe import generate_recipe_node
from app.agent.state import AgentState
from app.config import settings

logger = logging.getLogger(__name__)

#: The compiled graph as every caller outside this module sees it: state and
#: runtime context both pinned, so a caller that forgets `context=` on an
#: invocation is a type error rather than an `AttributeError` inside a node.
CueGraph = CompiledStateGraph[AgentState, CueContext]

GUARDRAIL = "guardrail"
GENERATE_RECIPE = "generate_recipe"
REFUSE = "refuse"


def build_graph() -> StateGraph[AgentState, CueContext]:
    """Build the agent's (uncompiled) chat loop.

    ```
    START -> guardrail ---(in_scope)---> generate_recipe -> END
                 |
                 +------(out_of_scope)-> refuse ---------> END
    ```

    This is the smallest graph that delivers the product's core loop: the
    user names a dish and gets its ingredient list back, and off-topic turns
    are turned away before any recipe model call.

    There is deliberately **no** static edge out of `guardrail`: it routes
    with a `Command`, which adds a *dynamic* edge, and a static edge
    alongside it would run both branches. `parse_recipe_photo_node` and
    `normalize_ingredients_node` stay unwired - text input only, for now.

    The graph is typed against `CueContext`, which carries the request-scoped
    handles nodes need but must never checkpoint - the database session above
    all. A fresh instance is supplied per `ainvoke`/`astream` by
    `chat.service.run_turn`; see `app/agent/context.py`.

    LangSmith tracing is not configured here: LangGraph/LangChain trace
    automatically once `LANGSMITH_TRACING`/`LANGSMITH_API_KEY`/`LANGSMITH_PROJECT`
    are real process env vars (see `main.py`'s `load_dotenv()` call) - no
    application code required.

    Returns:
        The uncompiled `StateGraph`; the caller compiles it (optionally with a
        checkpointer via `open_compiled_graph`).
    """
    builder: StateGraph[AgentState, CueContext] = StateGraph(
        AgentState, context_schema=CueContext
    )
    builder.add_node(GUARDRAIL, guardrail_node)
    builder.add_node(GENERATE_RECIPE, generate_recipe_node)
    builder.add_node(REFUSE, refuse_node)
    builder.add_edge(START, GUARDRAIL)
    builder.add_edge(GENERATE_RECIPE, END)
    builder.add_edge(REFUSE, END)
    return builder


def thread_config(thread_id: str) -> RunnableConfig:
    """Build the run config that keys the checkpointer for one session.

    Callers go through this rather than hand-writing the dict so a missing
    `thread_id` fails loudly here. A run with no thread id looks like it
    works and silently loses the conversation, which is far worse than an
    exception at the call site.

    Args:
        thread_id: `str(chat_session.id)` for the session being run.

    Returns:
        The `configurable` run config to pass alongside the state.

    Raises:
        ValueError: `thread_id` is empty, so the run would not be persisted.
    """
    if not thread_id:
        raise ValueError(
            "Cannot run the agent without a thread_id: the turn would run "
            "unpersisted and its history would be silently lost."
        )
    return {"configurable": {"thread_id": thread_id}}


def _checkpointer_conn_string() -> str:
    """Return a psycopg-compatible DSN for the checkpointer.

    `AsyncPostgresSaver` speaks psycopg (psycopg3), not SQLAlchemy's asyncpg
    driver, so the `+asyncpg` suffix is stripped from the shared
    `DATABASE_URL`.
    """
    return str(settings.DATABASE_URL).replace("postgresql+asyncpg", "postgresql")


@asynccontextmanager
async def open_compiled_graph(
    conn_string: str | None = None,
) -> AsyncIterator[CueGraph]:
    """Yield the compiled graph wired to a Postgres checkpointer.

    The checkpointer persists state against the same database as the app,
    keyed by `thread_id = str(chat_session.id)` supplied per-invocation in the
    run config. Callers use it as an async context manager so the underlying
    connection is closed on exit:

    ```python
    async with open_compiled_graph() as graph:
        await graph.ainvoke(state, {"configurable": {"thread_id": session_id}})
    ```

    Args:
        conn_string: Override DSN (used by tests); defaults to the app database.

    Yields:
        The compiled, checkpointed graph.
    """
    dsn = conn_string or _checkpointer_conn_string()
    async with AsyncPostgresSaver.from_conn_string(dsn) as checkpointer:
        await checkpointer.setup()
        yield build_graph().compile(checkpointer=checkpointer)
