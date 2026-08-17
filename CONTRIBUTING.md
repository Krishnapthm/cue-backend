# Contributing

Thank you for looking at Cue backend. This page tells you how to set up the
repository, how to make a change, and what the change must satisfy before it
lands.

`AGENTS.md` holds the full engineering rules for this repository, and a coding
agent reads it automatically. This page is the short version for a person.

## Set up

```bash
uv sync
cp .env.example .env    # then set DATABASE_URL and AUTH_FIREBASE_PROJECT_ID
uv run alembic upgrade head
uv run python scripts/setup_checkpointer.py
uv run uvicorn main:app --reload
```

[Getting started](docs/getting-started.md) explains each step, and explains
what to do when one fails.

## Commands

| Task | Command |
|---|---|
| Install | `uv sync` |
| Run the service | `uv run uvicorn main:app --reload` |
| Run the tests | `uv run pytest` |
| Run one test | `uv run pytest tests/cart/test_service.py::test_name -v` |
| Lint and format | `uv run ruff check --fix . && uv run ruff format .` |
| Type check | `uv run mypy .` |
| Apply migrations | `uv run alembic upgrade head` |

Run this before every commit:

```bash
uv run ruff check . && uv run mypy . && uv run pytest
```

## Structure

The repository is organized by domain, not by file type. One package for each
domain, under `app/`:

```
app/recipes/
  router.py        # API endpoints, HTTP concerns only
  schemas.py       # Pydantic request and response models
  models.py        # SQLAlchemy ORM models
  service.py       # business logic and database queries
  dependencies.py  # route dependencies
  config.py        # domain settings
  constants.py     # constants and error codes
  exceptions.py    # domain exceptions
```

Rules:

- A router delegates to a service. Business logic never lives in a route
  handler.
- There is no repository layer. Each service owns its queries.
- A cross-domain import names the module, for example
  `from app.recipes import service as recipe_service`. Never deep-import, and
  never use `import *`.

## Code style

- Type hints on every signature, parameter, and return value.
- `from __future__ import annotations` at the top of every module.
- Google-style docstrings on public functions.
- Pydantic models across every boundary. Never a raw dictionary.
- `pathlib.Path`, not `os.path`. F-strings only.
- The `logging` module, never `print()`.
- Ruff formats the code. The line length is 88.
- mypy runs in strict mode.

Write comments that explain why, not what. This repository documents the
reasoning behind a decision in the docstring, including the failure that
caused it. Keep that habit.

## Async rules

| The function does this | Use |
|---|---|
| Awaitable, non-blocking I/O | `async def` |
| Blocking I/O with no async client | `def`, so FastAPI uses the threadpool |
| A sync library inside an async route | `await run_in_threadpool(fn, *args)` |
| CPU-bound work over 50 ms | A worker process |

Never call blocking I/O inside `async def`. It freezes the event loop for
every request.

## Database

- SQLAlchemy 2.0 async only. Use `select()`, never `session.query()`.
- ORM only. No raw SQL, in any form.
- Shape the data in the query. Hydrate into Pydantic only for the response.
- Every schema change goes through Alembic. Never change a table by hand.
- Name a migration file by date and slug, for example
  `2026-08-17_add_budget_estimate.py`.

## Tests

The suite holds 661 tests, and it mirrors the `app/` structure.

- The suite starts a real, ephemeral Postgres with testcontainers, so Docker
  must run. Never SQLite, and never the development database.
- One Postgres instance serves the whole run, and no test truncates it. Scope
  an assertion query by user, and keep a unique value unique for each test.
- Use `httpx.AsyncClient` with `ASGITransport(app=app)`.
- Replace a dependency with `app.dependency_overrides`. Do not monkeypatch
  internals.
- Mock only authentication and external services, which means the model
  provider and the Swiggy MCP.
- No test calls a real model. `tests/conftest.py` removes the provider keys,
  so a missed stub fails loudly.
- Aim above 80 percent coverage of service-layer logic.

## Swiggy

Read the Swiggy Builders Club documentation before you write Swiggy code. The
machine-readable index is <https://mcp.swiggy.com/builders/llms.txt>, and the
human pages start at <https://mcp.swiggy.com/builders>.

Verify a tool name, a parameter, or an error code against those pages. Never
invent one. The fields inside a `data` payload are not specified, so read a
real response with `scripts/capture_search_products.py` instead of guessing.

## Commits

Format: `type(optional-scope): description`.

Types: `feat`, `fix`, `build`, `chore`, `ci`, `docs`, `style`, `refactor`,
`perf`, `test`. Use lowercase and the imperative, and end with no full stop.

Mark a breaking change with `!` before the colon, a `BREAKING CHANGE:` footer,
or both. Add a footer such as `Refs: #123` for a reference.

```
feat(agent): fan out one worker per needed ingredient
fix(instamart): parse Swiggy's real response envelope
docs(readme): describe the harness framing
```

## Pull requests

1. Branch from `main`, with `git switch -c <branch>`.
2. Make the change, and keep the documentation true. A new endpoint belongs in
   [the API reference](docs/reference/api.md), and a new setting belongs in
   [the configuration reference](docs/reference/configuration.md).
3. Run the three checks.
4. Add an entry to `CHANGELOG.md`, under `[Unreleased]`.
5. Open the pull request, and state what changed and why.

Start a bug fix by reproducing the bug end to end, as a user meets it. A fix
built on a guess repairs the wrong thing.
