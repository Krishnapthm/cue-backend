from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy

from app.agent.context import CueContext
from app.agent.nodes.guardrail import refuse_node
from app.agent.nodes.normalize import normalize_ingredients_node
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
NORMALIZE_INGREDIENTS = "normalize_ingredients"
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

#: Retry policy for the nodes that reach off-box - the recipe photo fetch and
#: parse, and the order-status lookup. A transient upstream failure is the
#: system's problem to retry, not the user's problem to read about; anything
#: that survives three attempts is a real failure and surfaces as one.
#:
#: Note this is retries of the *whole node*. Both nodes are safe to re-run:
#: neither writes anything, and `list_orders_throttled`'s floor means a retried
#: order-status node is served the cached list rather than re-hitting Swiggy.
NETWORK_RETRY = RetryPolicy(max_attempts=3)


def build_graph() -> StateGraph[AgentState, CueContext]:
    """Build the agent's (uncompiled) chat loop.

    ```
    START -> route_turn --(recipe)--------------> generate_recipe
                 |                                      ^    |
                 +-------(photo)--> parse_recipe_photo --+    v
                 |                                    normalize_ingredients
                 |                                            |
                 |                                            v
                 |                                           END
                 +-------(order_status)-> order_status ------> END
                 |
                 +-------(out_of_scope)-> refuse ------------> END
    ```

    Every turn enters through the router, which both labels the turn and
    picks its branch, so off-topic turns are turned away before any recipe
    model call.

    The two intake paths converge: a photo turn joins the text path at
    `generate_recipe` and gets a checklist and a cart like any other. On that
    turn `generate_recipe` renders the already-parsed recipe rather than
    generating a second one - see `_render_parsed_photo`.

    `generate_recipe -> normalize_ingredients` is a static edge: there is no
    routing decision to make, every recipe turn produces a checklist. The
    checklist is where `confirm_checklist` (CUE-90) will interrupt; until then
    the branch ends there.

    `parse_recipe_photo` and `order_status` both reach off-box and carry
    `NETWORK_RETRY`; see the note there.

    There is deliberately **no** static edge out of `route_turn`: it routes
    with a `Command`, which adds a *dynamic* edge, and a static edge
    alongside it would run both branches. The destinations are declared
    through the node's `Command[...]` return annotation instead.

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
    builder.add_node(
        PARSE_RECIPE_PHOTO, parse_recipe_photo_node, retry_policy=NETWORK_RETRY
    )
    builder.add_node(ORDER_STATUS, order_status_node, retry_policy=NETWORK_RETRY)
    builder.add_node(NORMALIZE_INGREDIENTS, normalize_ingredients_node)
    builder.add_node(REFUSE, refuse_node)
    builder.add_edge(START, ROUTE_TURN)
    builder.add_edge(PARSE_RECIPE_PHOTO, GENERATE_RECIPE)
    builder.add_edge(GENERATE_RECIPE, NORMALIZE_INGREDIENTS)
    builder.add_edge(NORMALIZE_INGREDIENTS, END)
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
