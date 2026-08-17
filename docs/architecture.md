# Architecture

This page explains how the harness fits together, and why it makes the
decisions it makes. It is background reading, not a procedure. For the
commands, read [Getting started](getting-started.md). For the exact contracts,
read the [reference pages](reference/api.md).

## The problem the harness solves

A user says "I want to cook biryani for four people". A language model can
write that recipe. It cannot buy the ingredients.

The distance between those two facts is the whole system:

- The model must not guess a product. A cart is real money and a real
  delivery.
- The model must not decide when to spend money. A person decides that.
- The model must not hold a database session. The state survives a pause of
  several days, and the session does not.
- The model must not stall the user. One turn makes many network calls.

The harness is the layer that supplies these properties. The model supplies
only the language work: classification, recipe knowledge, and short prose.

## The layers

```
Expo client (not in this repository)
        |  HTTPS, bearer token, server-sent events
+-------v--------------------------------------------------+
|  FastAPI service (app/main.py)                            |
|  routers -> dependencies -> domain services               |
+-------+-----------------------------+---------------------+
        |                             |
+-------v----------------+   +--------v--------------------+
|  Agent harness         |   |  Domain services            |
|  app/agent/            |   |  cart, tags, orders, pantry |
|  graph, nodes, state,  |   |  providers, addresses       |
|  context, model roles  |   |                             |
+-------+----------------+   +--------+--------------------+
        |                             |
        |         +-------------------v----------+
        +-------->|  Instamart tool layer        |
                  |  app/instamart/              |
                  +-------------------+----------+
                                      |  JSON-RPC 2.0
                              +-------v---------+
                              |  Swiggy MCP     |
                              +-----------------+

Postgres holds two separate stores:
  application tables (Alembic)  |  agent checkpoints (LangGraph)
```

The agent calls the same domain services that the HTTP routes call. A node
never speaks to Swiggy directly, and never writes SQL. This keeps one
implementation of each rule, whether a person or the agent triggers it.

## The eight harness concerns

The industry describes an agent as a model plus a harness. The list below maps
that idea onto this repository.

### 1. Tools

`app/instamart/client.py` speaks JSON-RPC 2.0 to the Swiggy MCP endpoint.
`app/instamart/service.py` wraps 11 tools: `get_addresses`, `create_address`,
`delete_address`, `search_products`, `update_cart`, `get_cart`, `checkout`,
`get_orders`, `get_order_details`, `your_go_to_items`, and `track_order`.

The tools are not exposed to the model as a tool-calling loop. The graph
decides which tool runs, and when. This is the central design decision of the
harness, and the next section explains it.

### 2. Control flow the model does not own

`app/agent/graph.py` builds a LangGraph state graph. The edges are static,
except for one router at the entry and one fan-out after the checklist.

A free tool-calling loop would let the model choose the order of the work.
Here the order is fixed, because each step has a correctness rule that a model
cannot be trusted to hold:

- Address selection happens before any cart work, because Swiggy binds a cart
  to an address.
- Availability verification happens before the harness offers a choice.
- The checklist pause happens before any search, because searches cost money
  and time.
- Checkout happens outside the graph entirely.

The model still does the parts that need language: it classifies the turn, it
writes the recipe, it reads the photo, and it writes one status sentence.

### 3. Memory

State has two stores, and they never merge.

| Store | Owner | Content | Keyed on |
|---|---|---|---|
| `chat_session` and `chat_message` | Alembic and the app | The display transcript the user reads | `chat_session.id` |
| LangGraph checkpoints | `scripts/setup_checkpointer.py` | `AgentState`, the graph's own memory | `thread_id`, which is `str(chat_session.id)` |

The two stores share a key but hold different data. The transcript is what the
user sees. The checkpoint is what the graph resumes from, which includes the
pending pause, the parsed recipe, and every ingredient match.

`AgentState` is a `TypedDict` in `app/agent/state.py`. Everything in it goes
to Postgres, so it holds only serializable values. The `matches` field carries
an `operator.add` reducer, because parallel workers write to it. Without the
reducer, the last worker to finish overwrites the others, and the user sees a
one-item checklist that looks correct.

`app/agent/graph.py` also pins an allowlist of types for the checkpoint
serializer. An explicit list keeps strict MsgPack safe, and stops a later
LangGraph release from turning a resumed chat into a failed request.

### 4. Runtime context

A database session cannot go into the checkpoint. It does not serialize, and
it does not survive a pause.

`CueContext` in `app/agent/context.py` is the answer. It is a frozen dataclass
that carries the request-scoped handles: the database session, the user id,
the chat session id, and the delivery address id. The caller supplies a fresh
instance for each invocation, and the graph type pins it, so a caller that
forgets it gets a type error rather than a failure inside a node.

This matters most at a pause. An `interrupt()` ends the invocation. The answer
arrives on a new HTTP request with a new database session. A context that is
supplied for each invocation is correct at that moment. A session hidden in
the state would be stale, or shared between users.

### 5. Guardrails

`route_turn` is the entry node and runs on every turn. It classifies the turn
into one of four intents and routes with a `Command`. An off-topic turn
reaches `refuse` before the harness spends a recipe model call.

The node wraps the user's text in delimiters and tells the classifier that
everything inside is data to judge, never instructions to follow. A prompt
injection is therefore something to classify, not something to obey.

The stream applies a second guardrail. `PROSE_NODES` is an allowlist of nodes
whose tokens may reach the user. Every other model call runs under
`with_structured_output`, so its tokens are fragments of an internal schema.
One of those fields is the guardrail reason, which the model writes from
attacker-influenced text. An allowlist stays silent until a node is declared
safe. A denylist would leak the first time somebody added a node.

### 6. Human checkpoints

The graph pauses twice, and both pauses use LangGraph `interrupt()`:

- **The source choice.** The harness finds one ready-made component that is
  available at the selected address, then asks the user whether to buy it or
  to cook it. If no verified component exists, the graph does not ask.
- **The ingredient checklist.** The harness shows the have and need rows, with
  the user's in-stock pantry items already ticked. The user's own answer
  always wins over the pantry seed.

The checklist pause sits immediately before the fan-out, and this position is
the point. Everything after it spends the user's money: one search for each
ingredient, then a real cart mutation. Nothing inside the fan-out may pause,
because the user already gave the instruction.

Checkout left the graph for the same reason. Swiggy's `checkout` creates and
confirms an order in one operation that is not safe to repeat, so it belongs
behind an explicit user action, not behind a model decision. `place_order` in
`app/cart/checkout.py` holds the logic. It never retries a transport failure.
It records the order as `unknown`, reads Swiggy's own recent orders, and hands
both back, because Swiggy offers no idempotency key. No HTTP route exposes
this service yet.

### 7. Failure handling

Three nodes reach off-box: the recipe photo fetch, the order-status lookup,
and each ingredient worker. All three carry `RetryPolicy(max_attempts=3)`, and
all three are safe to repeat, because none of them writes anything.

Retries alone are not enough, so the ingredient worker splits failures by
blast radius:

| Failure | Meaning | Result |
|---|---|---|
| `InstamartDomainError` | Swiggy ran the search and rejected it | One `unavailable` row, other rows still resolve |
| `InstamartAuthError` | The Swiggy token is dead | The turn ends, and the client shows a reconnect action |
| `InstamartTransportError` | Swiggy is unreachable after the retries | The turn ends, and the client shows a retry action |

The rule behind the table: a failure about one ingredient degrades one row. A
failure about the whole turn ends the turn once. The bad alternative is a cart
that silently misses real items, which the user finds only at the stove.

After the first byte of a stream, a status code is no longer available. Those
failures arrive as a typed `error` event that names the recovery action, and
the stream then closes cleanly. Anything that is not a known domain failure is
a bug, and it still raises.

### 8. Observability

LangGraph and LangChain trace to LangSmith as soon as `LANGSMITH_TRACING`,
`LANGSMITH_API_KEY`, and `LANGSMITH_PROJECT` are real process environment
variables. No application code is involved. `main.py` calls `load_dotenv()`
before it imports any application module, because the settings classes parse
`.env` into their own typed fields and never export it to the process
environment.

`scripts/agent_smoke.py` invokes the graph once and then queries the LangSmith
API, which proves the trace is queryable rather than merely visible.

The stream is the second observability surface, and it faces the user. Each
turn emits `stage` events for node changes, `match` events for each resolved
ingredient, `token` events for prose, `interrupt` events for a pause, and
exactly one `done` event.

## The model seam

No node names a model. A node asks for a role, and
`app/agent/config.py` maps the role to a model id:

| Role | Reason | Default |
|---|---|---|
| `ROUTER` | A four-way label on every turn. Cheap, no reasoning | `gpt-5.4-nano-2026-03-17` |
| `RECIPE` | Decides correctness. A wrong ingredient becomes a wrong order | `gpt-5.6-luna` |
| `VISION` | Reads a recipe photo into the same schema, with the same stakes | `gpt-5.6-luna` |
| `ORDER_STATUS` | Rewrites a validated payload into one sentence | `gpt-5.4-nano-2026-03-17` |
| `TITLE` | Names a chat session. Best-effort metadata | `gpt-4o-mini` |

`app/agent/providers.py` returns the model behind langchain-core's
`BaseChatModel`, so a node never imports `openai` or `anthropic`. A change of
provider is a change of configuration. The deterministic nodes,
`normalize_ingredients`, `report_cart`, and `refuse`, take no model at all.

## The stream

The blocking endpoint answers after several seconds of Swiggy calls and shows
nothing in between. The product resolves ingredient rows one at a time, and
the fan-out runs the searches in parallel, so the designed screen needs the
stream.

`stream_turn` in `app/chat/service.py` consumes three LangGraph stream modes
and translates every chunk into a typed event:

| Stream mode | Carries | Becomes |
|---|---|---|
| `messages` | Prose tokens from allowlisted nodes | `token` |
| `custom` | Per-ingredient payloads a worker writes | `match` |
| `updates` | Node changes, and the `__interrupt__` pause | `stage`, `interrupt` |

Raw LangGraph shapes never reach a client. They are an internal contract that
changes between library versions.

A stream drops whenever the phone goes to the background, so
`GET /chat/sessions/{id}/state` reads the pending interrupt back from the
checkpointer. There is no second column that tracks the pause, because a
second source of truth drifts from the first.

## Data and connections

The service holds two connection pools against the same Postgres instance:

| Pool | Driver | Used by | Settings |
|---|---|---|---|
| SQLAlchemy | asyncpg | Application tables | `DATABASE_POOL_SIZE`, `DATABASE_MAX_OVERFLOW` |
| Checkpointer | psycopg 3 | LangGraph checkpoints | `AGENT_CHECKPOINTER_POOL_MIN_SIZE`, `AGENT_CHECKPOINTER_POOL_MAX_SIZE` |

Size them together. Both count against the same Postgres `max_connections`.

The graph compiles once for each process, in the FastAPI lifespan. An earlier
version compiled it for each request, which opened one dedicated connection
for each call. That was tolerable for short blocking turns, and it failed as
soon as streams became long-lived: each open stream pinned a connection for
its whole duration. The pooled checkpointer borrows a connection for one
checkpoint read or write, then returns it, so an idle stream costs nothing.

Sharing one compiled graph between concurrent requests is safe by
construction. Every per-run input travels in the run config or in
`CueContext`, and nothing request-scoped is captured at compile time.

## Domain rules the harness enforces

These live in `app/cart/` and `app/matching/`, outside the model:

- **Variant selection.** `select_variant` picks the pack size and computes the
  quantity from the recipe amount.
- **Substitution.** When a product is out of stock, `rank_candidates` ranks
  purchasable candidates by pack-size distance, and `prefer_by_brand` favours
  the brands the user already buys. Every substitution is user-visible.
- **Cart composition.** `compose_cart` records a `cart_plan` row, pushes the
  lines to Swiggy, and reports the subtotal against the minimum order value.
- **Checkout.** `place_order` refuses an empty cart, refuses a second
  concurrent checkout, and stamps the pantry after the order is placed.

## What the harness does not do

Honest limits, as of today:

- There is no worker queue. The session title is the only background job, and
  it runs as an `asyncio` task that dies with the process. It is best-effort
  metadata, so a lost task costs the user nothing.
- `place_order` exists as a service, and no HTTP route calls it yet. The cart
  endpoints stop at the composed cart.
- There is no context compaction. A chat session grows until the model's
  context limit becomes the ceiling.
- There is no evaluation suite for the model nodes. Tests stub every model
  call, so they verify the harness, not the model's judgement.
- `parse_recipe_photo` reads `AgentState.image_object_path`, and the payload
  to state extraction for image messages is not wired yet.
- The harness supports one provider, Swiggy Instamart, and one country's
  grocery vocabulary.

## Further reading

- [Agent graph reference](reference/agent-graph.md) for each node and event
- [HTTP API reference](reference/api.md) for each endpoint
- [Configuration reference](reference/configuration.md) for each setting
- [Add a graph node](guides/add-a-graph-node.md) to extend the graph
