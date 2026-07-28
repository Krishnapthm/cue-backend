from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agent.context import CueContext
from app.agent.nodes.guardrail import refuse_node
from app.agent.nodes.order_status import order_status_node
from app.agent.nodes.recipe import generate_recipe_node, parse_recipe_photo_node
from app.agent.nodes.route import route_turn
from app.agent.state import AgentState
from app.config import settings

logger = logging.getLogger(__name__)

#: The compiled graph as every caller outside this module sees it: state and
#: runtime context both pinned, so a caller that forgets `context=` on an
#: invocation is a type error rather than an `AttributeError` inside a node.
CueGraph = CompiledStateGraph[AgentState, CueContext]

ROUTE_TURN = "route_turn"
GENERATE_RECIPE = "generate_recipe"
PARSE_RECIPE_PHOTO = "parse_recipe_photo"
ORDER_STATUS = "order_status"
REFUSE = "refuse"

#: Nodes whose model output is prose meant for the user, and whose tokens may
#: therefore be streamed to the client as they arrive.
#:
#: An allowlist, not a denylist, and deliberately so. Every other model call in
#: this graph runs under `with_structured_output`, so its token stream is JSON
#: fragments of an internal schema - including `GuardrailDecision.reason`,
#: which is attacker-influenceable text that must never reach the user. A
#: denylist would leak all of that the first time someone added a node and
#: forgot to update it; an allowlist stays silent until a node is declared
#: safe to stream.
PROSE_NODES: frozenset[str] = frozenset({ORDER_STATUS})


def build_graph() -> StateGraph[AgentState, CueContext]:
    """Build the agent's (uncompiled) chat loop.

    ```
    START -> route_turn --(recipe)-------> generate_recipe ---> END
                 |
                 +-------(photo)--------> parse_recipe_photo -> END
                 |
                 +-------(order_status)-> order_status ------> END
                 |
                 +-------(out_of_scope)-> refuse ------------> END
    ```

    Every turn enters through the router, which both labels the turn and
    picks its branch, so off-topic turns are turned away before any recipe
    model call. `normalize_ingredients_node` stays unwired.

    There is deliberately **no** static edge out of `route_turn`: it routes
    with a `Command`, which adds a *dynamic* edge, and a static edge
    alongside it would run both branches. The destinations are declared
    through the node's `Command[...]` return annotation instead.

    `order_status` is a deterministic stand-in until CUE-88 implements it; it
    is wired now because a `Command` destination that does not exist fails at
    compile time, so the route and its node must land together.

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
    builder.add_node(ROUTE_TURN, route_turn)
    builder.add_node(GENERATE_RECIPE, generate_recipe_node)
    builder.add_node(PARSE_RECIPE_PHOTO, parse_recipe_photo_node)
    builder.add_node(ORDER_STATUS, order_status_node)
    builder.add_node(REFUSE, refuse_node)
    builder.add_edge(START, ROUTE_TURN)
    builder.add_edge(GENERATE_RECIPE, END)
    builder.add_edge(PARSE_RECIPE_PHOTO, END)
    builder.add_edge(ORDER_STATUS, END)
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
