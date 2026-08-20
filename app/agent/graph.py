from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.agent.config import agent_settings
from app.agent.context import CueContext
from app.agent.nodes.cart import (
    COMPOSE_CART,
    REPORT_CART,
    compose_cart_node,
    report_cart_node,
)
from app.agent.nodes.confirm_checklist import confirm_checklist
from app.agent.nodes.guardrail import refuse_node
from app.agent.nodes.match_ingredient import (
    MATCH_INGREDIENT,
    MatchTask,
    fan_out,
    match_ingredient,
)
from app.agent.nodes.normalize import normalize_ingredients_node
from app.agent.nodes.order_status import order_status_node
from app.agent.nodes.recipe import generate_recipe_node, parse_recipe_photo_node
from app.agent.nodes.route import route_turn
from app.agent.nodes.scratch_choice import (
    choose_scratch_component,
    find_scratch_component,
)
from app.agent.nodes.title import schedule_title_node
from app.agent.schemas import (
    CartReport,
    CartReportItem,
    GeneratedRecipe,
    GuardrailDecision,
    IngredientStatus,
    MatchResult,
    NormalizedIngredient,
    RecipeStep,
    ScopeVerdict,
    ScratchChoice,
    ScratchComponent,
    TurnFailure,
    TurnFailureKind,
    TurnIntent,
)
from app.agent.state import AgentState
from app.cart.schemas import ComposeCartResult, MatchStatus
from app.config import settings

logger = logging.getLogger(__name__)

#: The compiled graph as every caller outside this module sees it: state and
#: runtime context both pinned, so a caller that forgets `context=` on an
#: invocation is a type error rather than an `AttributeError` inside a node.
CueGraph = CompiledStateGraph[AgentState, CueContext]

ROUTE_TURN = "route_turn"
GENERATE_RECIPE = "generate_recipe"
SCHEDULE_TITLE = "schedule_title"
PARSE_RECIPE_PHOTO = "parse_recipe_photo"
ORDER_STATUS = "order_status"
NORMALIZE_INGREDIENTS = "normalize_ingredients"
FIND_SCRATCH_COMPONENT = "find_scratch_component"
CHOOSE_SCRATCH_COMPONENT = "choose_scratch_component"
CONFIRM_CHECKLIST = "confirm_checklist"
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
#: parse, the order-status lookup, and each ingredient worker. A transient
#: upstream failure is the system's problem to retry, not the user's problem to
#: read about; anything that survives three attempts is a real failure and
#: surfaces as one.
#:
#: Note this is retries of the *whole node*. All three are safe to re-run:
#: none of them writes anything, `list_orders_throttled`'s floor means a
#: retried order-status node is served the cached list rather than re-hitting
#: Swiggy, and an ingredient worker only searches and ranks.
#:
#: It is worth being explicit about what this policy does *not* do, because
#: CUE-91 asks for both and they do not compose: a `RetryPolicy` re-raises once
#: its attempts are exhausted, so it cannot also be the thing that degrades one
#: worker's failure into an `unavailable` row. `add_node`'s `error_handler`
#: argument looks like the missing piece but is inert in langgraph 1.2.9 - it
#: is recorded on the node spec and read by nothing. Per-ingredient isolation
#: therefore lives in the worker itself, which catches the failures that are
#: about one ingredient and lets the ones that are about the whole turn
#: through; see `match_ingredient`.
NETWORK_RETRY = RetryPolicy(max_attempts=3)

# LangGraph will otherwise allow these application-owned checkpoint values only
# in permissive mode and warn on every recovery. Keeping this explicit makes
# strict MsgPack safe to enable and prevents a future LangGraph release from
# turning a resumed chat into a failed request.
_CHECKPOINTED_MSGPACK_TYPES: tuple[type[Any], ...] = (
    CartReport,
    CartReportItem,
    ComposeCartResult,
    GeneratedRecipe,
    GuardrailDecision,
    IngredientStatus,
    MatchResult,
    MatchStatus,
    NormalizedIngredient,
    RecipeStep,
    ScopeVerdict,
    ScratchChoice,
    ScratchComponent,
    TurnFailure,
    TurnFailureKind,
    TurnIntent,
)


def _checkpoint_serde() -> JsonPlusSerializer:
    """Return the explicit serializer allowlist for persisted agent state."""
    return JsonPlusSerializer(allowed_msgpack_modules=_CHECKPOINTED_MSGPACK_TYPES)


def build_graph() -> StateGraph[AgentState, CueContext]:
    """Build the agent's (uncompiled) chat loop.

    ```
    START -> route_turn --(recipe)--------------> generate_recipe
                 |                                      ^    |
                 +-------(photo)--> parse_recipe_photo --+    v
                 |                                    schedule_title
                 |                                            |
                 |                              find_scratch_component
                 |                                            |
                 |                              choose_scratch_component
                 |                                      (may pause)
                 |                                            |
                 |                                    normalize_ingredients
                 |                                            |
                 |                                            v
                 |                                    confirm_checklist  (pauses)
                 |                                            |
                 |                              (Send per NEED ingredient)
                 |                                            |
                 |                                            v
                 |                                    match_ingredient  (xN)
                 |                                            |
                 |                                            v
                 |                                      compose_cart
                 |                                            |
                 |                                            v
                 |                                       report_cart
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

    The recipe path is static through title scheduling, ready-made discovery,
    the optional scratch choice, normalization, and checklist confirmation.
    Discovery verifies availability against the selected address before it
    permits the choice card; a recipe with no verified component simply
    carries on.

    The scratch choice and `confirm_checklist` are both interrupts, so a
    compiled graph needs a checkpointer for every recipe turn. Checkout left
    the graph entirely (CUE-80), so no non-idempotent mutation sits before a
    pause.

    The edge out of `confirm_checklist` is conditional and returns `Send`s, one
    per ingredient the user still needs, so the workers run concurrently in a
    single super-step. It is the one place the graph fans out, and it sits
    *after* the interrupt on purpose: those searches are spent on the user's
    say-so, so nothing in the fan-out may pause to ask for more.

    `parse_recipe_photo`, `order_status` and `match_ingredient` all reach
    off-box and carry `NETWORK_RETRY`; see the note there.

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
    builder.add_node(SCHEDULE_TITLE, schedule_title_node)
    builder.add_node(
        PARSE_RECIPE_PHOTO, parse_recipe_photo_node, retry_policy=NETWORK_RETRY
    )
    builder.add_node(ORDER_STATUS, order_status_node, retry_policy=NETWORK_RETRY)
    builder.add_node(NORMALIZE_INGREDIENTS, normalize_ingredients_node)
    builder.add_node(FIND_SCRATCH_COMPONENT, find_scratch_component)
    builder.add_node(CHOOSE_SCRATCH_COMPONENT, choose_scratch_component)
    builder.add_node(CONFIRM_CHECKLIST, confirm_checklist)
    builder.add_node(
        MATCH_INGREDIENT,
        match_ingredient,
        input_schema=MatchTask,
        retry_policy=NETWORK_RETRY,
    )
    builder.add_node(REFUSE, refuse_node)
    builder.add_edge(START, ROUTE_TURN)
    builder.add_edge(PARSE_RECIPE_PHOTO, GENERATE_RECIPE)
    builder.add_edge(GENERATE_RECIPE, SCHEDULE_TITLE)
    builder.add_edge(SCHEDULE_TITLE, FIND_SCRATCH_COMPONENT)
    builder.add_edge(FIND_SCRATCH_COMPONENT, CHOOSE_SCRATCH_COMPONENT)
    builder.add_edge(CHOOSE_SCRATCH_COMPONENT, NORMALIZE_INGREDIENTS)
    builder.add_edge(NORMALIZE_INGREDIENTS, CONFIRM_CHECKLIST)
    builder.add_node(COMPOSE_CART, compose_cart_node)
    builder.add_node(REPORT_CART, report_cart_node)
    builder.add_conditional_edges(
        CONFIRM_CHECKLIST, fan_out, [MATCH_INGREDIENT, COMPOSE_CART]
    )
    builder.add_edge(MATCH_INGREDIENT, COMPOSE_CART)
    builder.add_edge(COMPOSE_CART, REPORT_CART)
    builder.add_edge(REPORT_CART, END)
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


def _checkpointer_conn_string(conn_string: str | None = None) -> str:
    """Return a psycopg-compatible DSN for the checkpointer.

    `AsyncPostgresSaver` speaks psycopg (psycopg3), not SQLAlchemy's asyncpg
    driver, so the `+asyncpg` suffix is stripped. An override is normalized the
    same way, so a caller can hand over whichever spelling of the URL it
    happens to be holding.

    Args:
        conn_string: Override DSN; defaults to the app database.

    Returns:
        The DSN psycopg will accept.
    """
    dsn = conn_string or str(settings.DATABASE_URL)
    return dsn.replace("postgresql+asyncpg", "postgresql")


async def setup_checkpointer(conn_string: str | None = None) -> None:
    """Provision the checkpointer's tables. **A deployment step, not a startup one.**

    This is DDL. It belongs beside `alembic upgrade head` in a deploy, run once
    against the database, and it deliberately does not run when the app boots:
    every process doing DDL on every start is a round-trip per boot at best and
    a migration race at worst. `scripts/setup_checkpointer.py` is the entry
    point; `open_compiled_graph` assumes the tables already exist.

    Args:
        conn_string: Override DSN; defaults to the app database.
    """
    async with AsyncPostgresSaver.from_conn_string(
        _checkpointer_conn_string(conn_string)
    ) as checkpointer:
        await checkpointer.setup()


@asynccontextmanager
async def open_compiled_graph(
    conn_string: str | None = None,
    *,
    min_size: int | None = None,
    max_size: int | None = None,
) -> AsyncIterator[CueGraph]:
    """Yield one compiled graph, backed by a checkpointer connection *pool*.

    Opened once per process from the FastAPI lifespan (see `app.main`), not
    once per request. A request-scoped graph recompiled the graph and opened a
    dedicated connection every call, which was tolerable for short `ainvoke`
    turns and is not once SSE streams are long-lived: every open stream pinned
    a connection for its whole duration, and a handful of concurrent users
    exhausted the pool. Pooled, a connection is borrowed for each checkpoint
    read or write and returned immediately, so open streams cost nothing while
    they are idle.

    ```python
    async with open_compiled_graph() as graph:
        await graph.ainvoke(state, thread_config(session_id), context=context)
    ```

    **The compiled graph is shared across concurrent requests, which is fine
    and intended.** Every per-run input travels in the run config (`thread_id`)
    or in `CueContext`, both supplied per invocation; nothing request-scoped is
    captured at compile time. That is exactly why `CueContext` exists - a
    session closed over by the compiled graph would be shared across users,
    which is the worst possible version of this bug.

    `setup()` is *not* called here; see `setup_checkpointer`.

    Args:
        conn_string: Override DSN (used by tests); defaults to the app database.
        min_size: Connections kept warm; defaults to the configured setting.
        max_size: Pool ceiling; defaults to the configured setting.

    Yields:
        The compiled, checkpointed graph. The pool closes on exit.
    """
    dsn = _checkpointer_conn_string(conn_string)
    async with AsyncConnectionPool(
        conninfo=dsn,
        min_size=agent_settings.CHECKPOINTER_POOL_MIN_SIZE
        if min_size is None
        else min_size,
        max_size=agent_settings.CHECKPOINTER_POOL_MAX_SIZE
        if max_size is None
        else max_size,
        # Both are required by `AsyncPostgresSaver`: it issues its own
        # transactions, and reads rows back as mappings.
        kwargs={"autocommit": True, "row_factory": dict_row},
        open=False,
    ) as pool:
        await pool.open(wait=True)
        # `row_factory=dict_row` above is what actually makes this pool hand
        # out dict rows, but it is passed as a runtime kwarg, so the
        # constructor's return type still says tuples. The cast states what the
        # kwargs already guarantee.
        checkpointer = AsyncPostgresSaver(
            cast("AsyncConnectionPool[AsyncConnection[dict[str, Any]]]", pool),
            serde=_checkpoint_serde(),
        )
        yield build_graph().compile(checkpointer=checkpointer)
