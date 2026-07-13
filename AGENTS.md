# AGENTS.md

## Project

Cue-backend is a uv workspace housing the FastAPI service (cue-api) and LangGraph agent (cue-agent) that power Cue, resolving a user's recipe intent into ingredient matches and a placed Swiggy Instamart order via the Swiggy MCP.

## Tech Stack

- **Language**: Python 3.13
- **Framework**: FastAPI (async-first)
- **Server**: Uvicorn
- **Database**: PostgreSQL via SQLAlchemy 2.0 async + asyncpg
- **Package manager**: uv (dependencies in `pyproject.toml`, locked in `uv.lock`)

## Commands

- Install: `uv sync`
- Dev: `uv run uvicorn main:app --reload`
- Test: `uv run pytest`
- Single test: `uv run pytest tests/recipes/test_service.py::test_name -v`
- Lint/format: `uv run ruff check --fix . && uv run ruff format .`
- Typecheck: `uv run mypy .`
- Build: not applicable (service, not a distributed package)

Run `uv run ruff check . && uv run mypy . && uv run pytest` before committing.

## Project Structure

Organize by domain (bounded context), not by file type. One package per domain under `app/`.

```
app/
├── recipes/            # example domain
│   ├── router.py       # API endpoints, HTTP concerns only
│   ├── schemas.py      # Pydantic request/response models
│   ├── models.py       # SQLAlchemy ORM models
│   ├── service.py      # business logic + DB queries
│   ├── dependencies.py # route dependencies
│   ├── config.py       # domain-scoped BaseSettings
│   ├── constants.py    # constants and error codes
│   └── exceptions.py   # domain exceptions
├── config.py           # global BaseSettings
├── exceptions.py       # global exception classes + handlers
├── database.py         # async engine + session factory
└── main.py             # FastAPI app + lifespan + router mounting
```

- Routers delegate to services; business logic never lives in route handlers.
- There is no separate repository layer: each domain's `service.py` owns its queries.
- Cross-domain imports use the explicit module name, e.g. `from app.recipes import service as recipe_service`. Never deep-import (`from app.recipes.service.match import ...`) and never `import *`.

## Coding Conventions

- Type hints on all function signatures, parameters and return types.
- `from __future__ import annotations` at the top of every module.
- Pydantic models for all API input/output; never pass raw dicts across boundaries.
- `pathlib.Path` instead of `os.path`; f-strings only (no `.format()` or `%`).
- Google-style docstrings on public functions.
- No `print()`; use the configured `logging` module.

## Async Rules

| Route/function does this              | Use                                            |
|---------------------------------------|------------------------------------------------|
| Awaitable non-blocking I/O            | `async def`                                    |
| Blocking I/O with no async client     | `def` (FastAPI runs it in the threadpool)      |
| Sync library inside an async route    | `await run_in_threadpool(fn, *args)`           |
| CPU-bound work (>50 ms)               | Offload to a worker process (Arq / Celery)     |

- Never call blocking I/O (`requests`, `time.sleep`, sync DB drivers, `open()`) inside `async def`; it freezes the event loop for every request. Use `httpx.AsyncClient`, `asyncio.sleep`, and async drivers.
- Don't make routes sync "just because": the default threadpool is 40 threads and saturating it slows all sync routes.

## Pydantic

- Use built-in types and constraints (`EmailStr`, `AnyUrl`, `Field(min_length=..., ge=...)`, `StrEnum`) before writing custom validators.
- Don't combine a constraint with a contradicting default like `Field(ge=18, default=None)`; a field is either required or `T | None`.
- `json_encoders` is removed in Pydantic v2; use `@field_serializer` or `Annotated[T, PlainSerializer(...)]`.
- One `BaseSettings` subclass per domain (with `env_prefix`), not one giant settings object for the whole app.

## Dependencies

- Use the `Annotated` form: `user: Annotated[User, Depends(get_current_user)]`, not the default-argument form.
- Dependencies should validate, not just inject: load the object, raise the domain exception if missing, return it.

```python
async def valid_recipe_id(recipe_id: UUID4) -> Recipe:
    recipe = await service.get_by_id(recipe_id)
    if not recipe:
        raise RecipeNotFound()
    return recipe
```

- Chain dependencies for reuse (e.g. `valid_owned_recipe` depends on `valid_recipe_id` + auth).
- Dependencies are cached per request: the same `Depends(x)` used five times runs once.
- Prefer `async def` dependencies; sync ones burn a threadpool slot.

## Database

- SQLAlchemy 2.0 async API only: `create_async_engine`, `async_sessionmaker`, `AsyncSession`, `select()` (never `session.query()`).
- ORM only, under no circumstance raw SQL: no `text()`, no raw SQL strings, no driver-level queries. Everything is expressed through SQLAlchemy constructs.
- Database-first, Pydantic-second: do joins, aggregation, and shaping in the query (via SQLAlchemy expressions); hydrate into Pydantic only for response validation.
- Naming: `lower_case_snake`; singular tables (`recipe`, `order_item`); group with a prefix (`order_item`, `order_event`); `_at` for datetimes, `_date` for dates; the same FK column name everywhere it appears.
- Set an explicit index naming convention on `MetaData` so Alembic autogenerate stays deterministic.
- All schema changes go through Alembic; never modify tables manually. Migrations must be static and reversible. Use the async template (`alembic init -t async migrations`) and date-slug filenames (`2026-07-11_add_recipe_idx.py`).

## Error Handling

- Custom exception classes per domain in `exceptions.py`; services raise domain exceptions, a global handler maps them to responses.
- Never catch bare `Exception` around a route body; it hides bugs and turns 500s into silent 200s. Catch specific classes and raise `HTTPException` with a meaningful status.
- Never leak unhandled-exception details in production responses.

## Background Work

- `BackgroundTasks` only for sub-second, in-process, droppable work; it runs in the same worker with no retry and dies with the process.
- Anything you'd page on (retries, scheduling, CPU-heavy, minutes-long) goes to a worker queue (Arq / Celery).

## Testing

- Async client from day one: `httpx.AsyncClient` + `ASGITransport(app=app)`. Never `async_asgi_testclient` (unmaintained).
- Swap dependencies with `app.dependency_overrides[dep] = fake`; don't monkeypatch internals.
- Use a real, ephemeral PostgreSQL for integration tests (testcontainers or a dedicated test database); never SQLite substitutes and never the dev database. Mock only auth and external services (e.g. Swiggy MCP).
- `tests/` mirrors the `app/` domain structure; shared fixtures in `conftest.py`, factory fixtures for test data.
- Aim for >80% coverage on service-layer business logic.

## API Documentation

- Disable `/docs` outside local/staging by setting `openapi_url=None` based on `settings.ENVIRONMENT`.
- Document endpoints fully: `response_model`, `status_code`, `summary`, `tags`, and error `responses`.
- Don't return a Pydantic model *and* set `response_model=` to the same class; the model gets validated twice. Return the raw row/dict and let `response_model` validate.

## Never

- Commit secrets
- Touch .env
- Blocking calls inside `async def`
- `python-jose` for JWT (unmaintained); use PyJWT (`import jwt`)
- Raw dicts across API boundaries
- Business logic in route handlers
- Raw SQL in any form (`text()`, string queries, driver-level execution); ORM only
- Manual schema changes outside Alembic

## When stuck

Ask a clarifying question or propose a short plan first.
