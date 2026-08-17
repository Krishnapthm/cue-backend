# Cue backend

Cue backend is an agent harness. It turns a sentence such as "I want to cook
palak paneer for four people" into a verified Swiggy Instamart cart, and then
into a placed order.

The language model is one part of this repository. The rest is the harness:
the tools the model may call, the state it keeps, the points where it stops
and asks the user, and the traces that show what it did.

Status: this is a working prototype and a portfolio piece. It is not a
released package, and it has no public deployment. See
[Limitations](#limitations).

## Agent = model + harness

An agent harness is the software layer around a model that lets the model act.
It supplies tools, memory, guardrails, human checkpoints, and observability.
The model supplies the reasoning. This repository holds the harness for one
narrow task: recipe to groceries.

| Harness concern | Where it lives in this repository |
|---|---|
| Tools | `app/instamart/` wraps 11 Swiggy Instamart MCP tools |
| Control flow | `app/agent/graph.py` builds a LangGraph state graph |
| Memory | A Postgres checkpointer keys agent state on the chat session id |
| Guardrails | `route_turn` classifies each turn and refuses off-topic ones |
| Human checkpoints | Two `interrupt()` pauses ask the user before money moves |
| Model seam | `app/agent/providers.py` maps a node role to a model id |
| Observability | LangSmith traces, plus typed server-sent events per turn |
| Transport | A FastAPI service streams each turn to the mobile client |

The [architecture guide](docs/architecture.md) explains each row, and explains
why the harness makes these decisions instead of the model.

## Quickstart

You need Python 3.13, [uv](https://docs.astral.sh/uv/), and a Postgres
database.

```bash
uv sync
cp .env.example .env          # then set DATABASE_URL and AUTH_FIREBASE_PROJECT_ID
uv run alembic upgrade head
uv run python scripts/setup_checkpointer.py
uv run uvicorn main:app --reload
```

The service starts on `http://127.0.0.1:8000`. Send a request to the liveness
probe:

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok","environment":"local"}
```

The interactive API documentation is at `http://127.0.0.1:8000/docs` in the
`local` and `staging` environments. Production disables it.

For the full first run, which includes a Swiggy account link and a real chat
turn, read [Getting started](docs/getting-started.md).

## What the harness does in one turn

1. The client opens `GET /chat/sessions/{id}/stream` with the user's message.
2. `route_turn` classifies the turn: recipe, photo, order status, or refusal.
3. `generate_recipe` produces a structured recipe with a fixed schema.
4. `find_scratch_component` looks for one ready-made alternative, such as a
   sauce base, and asks the user to choose. This is the first pause.
5. `normalize_ingredients` reshapes the recipe into have and need rows, and
   pre-ticks the staples that the user's pantry already holds.
6. `confirm_checklist` shows the checklist and pauses. This is the second
   pause, and it is the last point before the harness spends money.
7. One worker per needed ingredient searches Instamart in parallel. Each
   worker streams its result as a `match` event.
8. `compose_cart` writes the plan and pushes it to the Swiggy cart.
9. `report_cart` renders the closing card from the cart that Swiggy holds.

Checkout is not in the graph, so no operation that spends money sits behind a
model decision. `app/cart/checkout.py` holds that logic as a service. No HTTP
route exposes it yet.

## Documentation

| Page | Purpose |
|---|---|
| [Getting started](docs/getting-started.md) | Run the service and complete one turn |
| [Architecture](docs/architecture.md) | How the harness fits together, and why |
| [Agent graph reference](docs/reference/agent-graph.md) | Nodes, state, and stream events |
| [HTTP API reference](docs/reference/api.md) | Every endpoint, with its status codes |
| [Configuration reference](docs/reference/configuration.md) | Every environment variable |
| [Add a graph node](docs/guides/add-a-graph-node.md) | Extend the agent safely |
| [Link a Swiggy account](docs/guides/link-a-swiggy-account.md) | Complete the OAuth flow |
| [Contributing](CONTRIBUTING.md) | Set up, test, and submit a change |
| [Changelog](CHANGELOG.md) | What changed, and when |

## Repository map

```
app/
  agent/          The LangGraph harness: graph, nodes, state, context, models
  instamart/      Swiggy Instamart MCP client and tool wrappers
  chat/           Chat sessions, the turn runner, and the SSE stream
  cart/           Variant selection, cart composition, and checkout
  matching/       Out-of-stock substitution ranking
  providers/      Swiggy OAuth 2.1 and PKCE link flow
  auth/           Firebase ID token verification
  pantry/  tags/  Pantry items, and NFC tag to product binding
  orders/         Order history and live tracking
  addresses/      Swiggy delivery addresses
  health/         Liveness and database probes
  models/         SQLAlchemy ORM models
migrations/       Alembic migrations
scripts/          Deployment and diagnostic scripts
tests/            Pytest suite, mirroring the app structure
```

## Technology

- Python 3.13, FastAPI, and Uvicorn
- LangGraph 1.2 for the agent graph, with a Postgres checkpointer
- SQLAlchemy 2.0 async and asyncpg for application data
- Alembic for schema migrations
- uv for dependency management
- Ruff, mypy in strict mode, and pytest

## Swiggy MCP

Every product, cart, and order operation goes through the Swiggy Builders Club
MCP. `app/instamart/` is the only module that speaks to it.

Point your coding agent at the machine-readable index first. It is the
canonical entry point:

- Index: <https://mcp.swiggy.com/builders/llms.txt>
- Full text: <https://mcp.swiggy.com/builders/llms-full.txt>
- Any page as Markdown: append `.md` to a docs URL, for example
  <https://mcp.swiggy.com/builders/docs/start/authenticate.md>. A section
  index takes `/index.md`, for example
  <https://mcp.swiggy.com/builders/docs/reference/instamart/index.md>

The same pages for a human reader:

- Documentation home: <https://mcp.swiggy.com/builders>
- Instamart tool schemas: <https://mcp.swiggy.com/builders/docs/reference/instamart>
- Error codes: <https://mcp.swiggy.com/builders/docs/reference/errors>
- Authentication: <https://mcp.swiggy.com/builders/docs/start/authenticate>

Verify a tool name, a parameter, or an error code against these pages before
you write Swiggy code. `app/instamart/schemas.py` shows why: the response
fields inside `data` are not specified, so `Product` once read `name` and
`variants` while Swiggy sent `displayName` and `variations`. The tests stayed
green and every tag resolution returned `unresolved`. Use
`scripts/capture_search_products.py` to read the real payload instead of
guessing.

## Limitations

The documentation states only what the code does. These points are true today:

- The service depends on the Swiggy Builders Club MCP. Without a linked
  Swiggy account, every product, cart, and order endpoint fails.
- The test suite starts a real Postgres container with testcontainers. You
  must run Docker to run the tests.
- No test calls a real model. The suite removes the provider API keys.
- The repository has no license file. Read the terms before you reuse it.
- The repository contains the backend only. The Expo client is not here.

## Credits

Cue is built by [Krishnapthm](https://github.com/Krishnapthm).
