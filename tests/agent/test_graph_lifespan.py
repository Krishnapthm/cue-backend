"""One compiled graph per process, backed by a pool (CUE-93).

The thing these tests actually defend is a capacity bug rather than a wrong
answer: the graph used to be compiled per request, opening a dedicated
checkpointer connection each time, so a long-lived SSE stream pinned one
connection for its whole duration and a handful of concurrent users exhausted
the database.

`setup()` is deliberately absent from the request path, which is why the suite
provisions the checkpoint tables the way a deploy does - see `postgres_url` in
`tests/conftest.py`.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Request
from httpx import ASGITransport
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.agent import graph as graph_module
from app.agent.schemas import (
    GeneratedRecipe,
    MatchResult,
    RecipeIngredient,
    RecipeStep,
)
from app.cart.schemas import MatchStatus
from app.chat.dependencies import agent_graph


@pytest.fixture
def dsn(postgres_url: str) -> str:
    return postgres_url.replace("postgresql+asyncpg", "postgresql")


# --- the pool ---------------------------------------------------------------


async def test_more_concurrent_readers_than_connections_all_complete(
    dsn: str,
) -> None:
    """The regression this issue exists for, at the layer that caused it.

    Eight concurrent checkpointer reads against a pool of two. Under the old
    connection-per-request design each caller held its own connection for the
    length of its request; pooled, a connection is borrowed per operation and
    handed straight back, so two serve eight. A deadlock or a timeout here
    means the checkpointer has started holding connections again.
    """
    async with graph_module.open_compiled_graph(dsn, min_size=1, max_size=2) as graph:
        threads = [str(uuid.uuid4()) for _ in range(8)]

        async def _run(thread_id: str) -> Any:
            return await graph.aget_state(graph_module.thread_config(thread_id))

        results = await asyncio.wait_for(
            asyncio.gather(*(_run(thread) for thread in threads)), timeout=30
        )

    assert len(results) == 8


async def test_a_pool_smaller_than_the_work_still_serves_every_thread(
    dsn: str,
) -> None:
    """Reads and writes interleaved across more threads than the pool holds."""
    async with graph_module.open_compiled_graph(dsn, min_size=1, max_size=2) as graph:
        threads = [str(uuid.uuid4()) for _ in range(6)]

        async def _write(thread_id: str) -> None:
            config = graph_module.thread_config(thread_id)
            await graph.aupdate_state(config, {"session_id": thread_id, "user_id": 1})

        await asyncio.wait_for(
            asyncio.gather(*(_write(thread) for thread in threads)), timeout=30
        )

        states = [
            await graph.aget_state(graph_module.thread_config(thread))
            for thread in threads
        ]

    assert [state.values["session_id"] for state in states] == threads


async def test_two_users_threads_stay_isolated_on_one_shared_graph(
    dsn: str,
) -> None:
    """The compiled graph is shared; the state behind it must not be.

    Everything per-run travels in the run config, so one graph serving two
    users is safe - this is the test that says so out loud.
    """
    alice, bob = str(uuid.uuid4()), str(uuid.uuid4())

    async with graph_module.open_compiled_graph(dsn, min_size=1, max_size=4) as graph:
        await graph.aupdate_state(
            graph_module.thread_config(alice), {"user_id": 1, "session_id": alice}
        )
        await graph.aupdate_state(
            graph_module.thread_config(bob), {"user_id": 2, "session_id": bob}
        )

        alice_state = await graph.aget_state(graph_module.thread_config(alice))
        bob_state = await graph.aget_state(graph_module.thread_config(bob))

    assert alice_state.values["user_id"] == 1
    assert bob_state.values["user_id"] == 2


async def test_a_pause_survives_being_resumed_through_a_second_pool(
    dsn: str,
) -> None:
    """Interrupt and resume still work across processes, which is the point.

    Two pools, two compiled graphs, one `thread_id` - a redeploy between the
    question and the answer. Pooling the checkpointer must not cost this.
    """
    thread_id = str(uuid.uuid4())
    config = graph_module.thread_config(thread_id)

    async with graph_module.open_compiled_graph(dsn, min_size=1, max_size=2) as graph:
        await graph.aupdate_state(config, {"session_id": thread_id, "user_id": 7})

    async with graph_module.open_compiled_graph(dsn, min_size=1, max_size=2) as graph:
        state = await graph.aget_state(config)

    assert state.values["user_id"] == 7


# --- setup is a deployment step --------------------------------------------


async def test_opening_the_graph_does_no_ddl(
    dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`setup()` on every request was a DDL round-trip per call.

    The persistence skill is explicit that it belongs to deployment. Asserting
    it is *not* called is the only way to keep it out of the request path.
    """
    called = False

    async def _tripwire(_self: Any) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(AsyncPostgresSaver, "setup", _tripwire, raising=True)

    async with graph_module.open_compiled_graph(dsn, min_size=1, max_size=2):
        pass

    assert called is False


async def test_setup_checkpointer_is_idempotent(dsn: str) -> None:
    """The deploy step runs on every deploy, against a database that has it."""
    await graph_module.setup_checkpointer(dsn)
    await graph_module.setup_checkpointer(dsn)


def test_the_dsn_is_normalised_whichever_spelling_it_arrives_in() -> None:
    """`AsyncPostgresSaver` speaks psycopg; SQLAlchemy's URL says `+asyncpg`."""
    normalised = graph_module._checkpointer_conn_string(
        "postgresql+asyncpg://u:p@host/db"
    )

    assert normalised == "postgresql://u:p@host/db"


def test_checkpointer_serializer_explicitly_allows_agent_state_types(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A resumed cart must not depend on LangGraph's permissive MsgPack mode."""
    result = MatchResult(ingredient_name="milk", status=MatchStatus.MATCHED)
    serde = graph_module._checkpoint_serde()

    with caplog.at_level("WARNING", logger="langgraph.checkpoint.serde.jsonplus"):
        restored = serde.loads_typed(serde.dumps_typed(result))

    assert restored == result
    assert "Deserializing unregistered type" not in caplog.text


def test_checkpointer_serializer_round_trips_a_recipe_with_steps(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`RecipeStep` is on the allowlist, so a resumed cooking session survives.

    Nothing in the test suite fails when a new checkpointed type is missing
    from `_CHECKPOINTED_MSGPACK_TYPES` - it fails in production, on the resume,
    which is the one moment the user has already committed to the turn. This is
    that test for CUE-116's `steps`.
    """
    recipe = GeneratedRecipe(
        dish_name="paneer butter masala",
        estimated_time_minutes=35,
        ingredients=[RecipeIngredient(name="paneer", quantity=250, unit="g")],
        method_summary="Simmer the gravy, fold in paneer.",
        steps=[
            RecipeStep(
                title="Simmer the gravy",
                instructions=["Simmer until it thickens."],
                duration_seconds=900,
            ),
            RecipeStep(title="Fold in the paneer", instructions=["Fold gently."]),
        ],
        servings=2,
        difficulty="Easy",
    )
    serde = graph_module._checkpoint_serde()

    with caplog.at_level("WARNING", logger="langgraph.checkpoint.serde.jsonplus"):
        restored = serde.loads_typed(serde.dumps_typed(recipe))

    assert restored == recipe
    assert restored.steps[0].duration_seconds == 900
    assert restored.steps[1].duration_seconds is None
    assert "Deserializing unregistered type" not in caplog.text


# --- the dependency ---------------------------------------------------------


def _lifespan_app(dsn: str) -> FastAPI:
    """A tiny app wired exactly as `app.main` wires the real one."""

    @asynccontextmanager
    async def lifespan(fastapi_app: FastAPI) -> AsyncIterator[None]:
        async with graph_module.open_compiled_graph(
            dsn, min_size=1, max_size=2
        ) as graph:
            fastapi_app.state.agent_graph = graph
            yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/graph-id")
    async def _graph_id(request: Request) -> dict[str, int]:
        return {"id": id(agent_graph(request))}

    return app


async def test_every_request_is_handed_the_same_compiled_graph(dsn: str) -> None:
    """One graph per process, not per request - the whole change in one line."""
    app = _lifespan_app(dsn)

    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            first = (await client.get("/graph-id")).json()["id"]
            second = (await client.get("/graph-id")).json()["id"]

    assert first == second


async def test_the_dependency_fails_loudly_without_a_lifespan() -> None:
    """Compiling one here would silently reintroduce the bug under load."""
    app = FastAPI()
    transport = ASGITransport(app=app)

    class _Request:
        def __init__(self, fastapi_app: FastAPI) -> None:
            self.app = fastapi_app

    with pytest.raises(RuntimeError, match="without running its lifespan"):
        agent_graph(_Request(app))  # type: ignore[arg-type]

    await transport.aclose()
