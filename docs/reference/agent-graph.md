# Agent graph reference

This page documents the LangGraph graph in `app/agent/`: the nodes, the state,
the pauses, and the stream events. For the reasons behind the design, read
[Architecture](../architecture.md).

## The graph

`build_graph()` in `app/agent/graph.py` returns the uncompiled `StateGraph`.
`open_compiled_graph()` compiles it with the Postgres checkpointer and yields
it. The FastAPI lifespan opens it once for each process.

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
             |                                       report_cart -> END
             +-------(order_status)-> order_status ------> END
             |
             +--(cooking_question)-> answer_cooking_question -> END
             |
             +-------(small_talk)---> small_talk --------> END
             |
             +-------(out_of_scope)-> refuse ------------> END
```

`route_turn` has no static outgoing edge. It routes with a `Command`, which
adds a dynamic edge. A static edge beside it would run both branches. The node
declares its destinations through its `Command[...]` return annotation.

## Nodes

| Node | Model role | Purpose |
|---|---|---|
| `route_turn` | `ROUTER` | Classify the turn into one of six intents, and route it |
| `generate_recipe` | `RECIPE` | Produce a structured recipe for the named dish |
| `parse_recipe_photo` | `VISION` | Read an uploaded photo into the same recipe schema |
| `schedule_title` | `TITLE` | Start one best-effort attempt to name the session |
| `find_scratch_component` | none | Find one ready-made component that the address stocks |
| `choose_scratch_component` | none | Pause for the source choice, then render the list |
| `normalize_ingredients` | none | Reshape the recipe into have and need rows |
| `confirm_checklist` | none | Pause, and ask which ingredients the user owns |
| `match_ingredient` | none | Resolve one ingredient against Instamart, in parallel |
| `compose_cart` | none | Record the plan, and push it to the Swiggy cart |
| `report_cart` | none | Render the closing card from the cart Swiggy holds |
| `order_status` | `ORDER_STATUS` | Answer "where is my order" from the real order list |
| `answer_cooking_question` | `COOKING` | Answer a question about the step the user is on |
| `small_talk` | `SMALL_TALK` | Answer a thank-you or a greeting in one short line |
| `refuse` | none | Append a fixed refusal message. No model runs |

`app/agent/config.py` maps each role to a model id. See the
[configuration reference](configuration.md#agent).

### Intents and destinations

`route_turn` returns one of six intents:

| Intent | Destination |
|---|---|
| `recipe` | `generate_recipe` |
| `photo` | `parse_recipe_photo` |
| `order_status` | `order_status` |
| `cooking_question` | `answer_cooking_question` |
| `small_talk` | `small_talk` |
| `out_of_scope` | `refuse` |

`cooking_question` is offered to the classifier only on a turn that carries an
`active_step_index` *and* whose session already holds a recipe. On any other
turn the label is not described in the prompt, and a model that returns it
anyway fails closed to `refuse`.

`small_talk` takes the pleasantries - thanks, praise, greetings - that would
otherwise fall into `out_of_scope` and be answered with the refusal. It writes
nothing but `messages`, and its reply is capped at one short line.

The classifier receives the user's turn between delimiters, and the prompt
states that everything inside is data to judge, never instructions to follow.

### Retries

`parse_recipe_photo`, `order_status`, and `match_ingredient` carry
`RetryPolicy(max_attempts=3)`. All three reach off-box, and none of them
writes anything, so a repeat is safe.

A `RetryPolicy` raises again once it exhausts its attempts. It therefore
cannot also degrade one worker's failure into an `unavailable` row. That
isolation lives inside `match_ingredient`:

| Exception | Blast radius | Result |
|---|---|---|
| `InstamartDomainError` | One ingredient | An `unavailable` row |
| `InstamartAuthError` | The whole turn | The turn ends, with a reconnect action |
| `InstamartTransportError` | The whole turn | The turn ends, with a retry action |

### The fan-out

`confirm_checklist` leaves through a conditional edge that returns `Send`
objects, one for each ingredient the user still needs. The workers run
together in a single super-step.

Each worker receives a `MatchTask`, which holds one `NormalizedIngredient` and
nothing else. A small payload keeps the checkpoint for each parallel branch
small.

Each worker writes its result with `get_stream_writer()`, which reaches the
client as a `match` event. The workers finish out of order. Each event carries
the ingredient name as a stable key, so the client fills its row in place.

## State

`AgentState` in `app/agent/state.py` is a `TypedDict`. The checkpointer writes
all of it to Postgres, so every value must serialize.

| Field | Type | Notes |
|---|---|---|
| `session_id` | `str` | Also the checkpointer `thread_id` |
| `user_id` | `int` | The owning Cue user |
| `messages` | `list[BaseMessage]` | Uses the `add_messages` reducer, so nodes append |
| `recipe` | `GeneratedRecipe` or null | Optional |
| `guardrail` | `GuardrailDecision` or null | Kept for logs and traces only |
| `image_object_path` | `str` or null | The uploaded photo's storage path |
| `have_marks` | `set[str]` | Ingredients marked as owned, spelled as the recipe spells them |
| `normalized_ingredients` | list | The have and need rows |
| `scratch_component` | `ScratchComponent` or null | The one verified ready-made option |
| `scratch_choice` | `ScratchChoice` or null | The user's answer, so a replay does not ask again |
| `turn_intent` | `TurnIntent` | The branch this turn took |
| `matches` | list | Uses the `operator.add` reducer |
| `cart_plan_id` | `int` or null | The recorded plan row |
| `compose_result` | `ComposeCartResult` or null | Subtotal and minimum-order verdict |
| `cart_report` | `CartReport` or null | The closing card |
| `failure` | `TurnFailure` or null | A failure the user can act on |
| `title_attempted` | `bool` | Set when the title task starts, so later turns skip it |

Two reducers matter:

- `messages` uses `add_messages`. Node returns append to the transcript, and
  upsert by id, instead of overwriting it.
- `matches` uses `operator.add`. Parallel workers write to it. Without the
  reducer, the last worker to finish overwrites every other result, and the
  user sees a plausible one-item checklist rather than an error.

`have_marks` has two sources, with a strict order. The user's own answer wins.
Only when there is none does `normalize_ingredients` seed it from the user's
in-stock pantry items.

## Runtime context

`CueContext` in `app/agent/context.py` carries what a node needs but must
never checkpoint.

| Attribute | Type | Purpose |
|---|---|---|
| `session` | `AsyncSession` | The request's database session, so node writes join the request's unit of work |
| `user_id` | `int` | The Cue user |
| `chat_session_id` | `UUID` | The chat session. Its `str()` is the `thread_id` |
| `address_id` | `str` | The Swiggy address the cart binds to |

`chat.service.run_turn` builds a fresh instance for each invocation. The
compiled graph type pins `CueContext`, so an invocation that omits it is a
type error.

## Pauses

The graph pauses with LangGraph `interrupt()`. A compiled graph therefore
needs a checkpointer for every recipe turn.

### The source choice

Payload, from `ScratchChoiceInterrupt`:

```json
{
  "ui": "scratch_choice",
  "dish_name": "Chicken biryani",
  "component_name": "biryani masala",
  "ready_made_name": "Everest Biryani Masala 50 g",
  "options": [
    {"id": "ready_made", "label": "..."},
    {"id": "from_scratch", "label": "..."}
  ]
}
```

Answer, from `ScratchChoiceDecision`:

```json
{"choice": "from_scratch"}
```

The node asks only when `find_scratch_component` verified a component against
the selected address. A recipe with no verified component passes straight
through.

### The ingredient checklist

Payload, from `ChecklistInterrupt`:

```json
{
  "ui": "checklist",
  "items": [
    {"name": "spinach", "quantity": 500, "unit": "g", "have": false},
    {"name": "salt", "quantity": null, "unit": null, "have": true}
  ]
}
```

Answer, from `ChecklistDecision`:

```json
{"have": ["salt", "oil"]}
```

The answer is the consent that authorizes a change to the user's Swiggy cart,
so the harness validates it rather than reading a raw dictionary.

### How a client answers

Both answers arrive as a message with `kind` set to `checklist`:

```
POST /chat/sessions/{session_id}/messages
{"role": "user", "kind": "checklist", "payload": {"have": ["salt"]}}
```

The service reads the pending interrupt from the checkpointer, then picks the
schema from the payload's `ui` field. It resumes with `Command(resume=...)`.
A plain state dictionary would not raise. It would start a new run and leave
the session stuck.

A payload that carries neither `have` nor `choice` returns 422. An answer with
no pause open returns 422 as well.

## Stream events

`GET /chat/sessions/{id}/stream` sends named server-sent events. `app/chat/sse.py`
encodes them, and the stream also sends `: keepalive` comment lines.

```
event: match
data: {"event":"match","ingredient_name":"spinach","status":"matched",...}

```

| Event | Source stream mode | Payload |
|---|---|---|
| `token` | `messages` | `text` |
| `match` | `custom` | One ingredient result, see below |
| `stage` | `updates` | `node`, the name of the node that just ran |
| `interrupt` | `updates` | `id`, `payload` |
| `error` | Domain failure | `code`, `message`, `action` |
| `done` | End of turn | `reply`, `message_id`, `interrupted` |

A turn always ends with exactly one `done` event.

### token

Only nodes in the `PROSE_NODES` allowlist emit tokens. Today that is
`order_status` alone. Every other model call runs under
`with_structured_output`, and its tokens are fragments of an internal schema.
One of those fields holds attacker-influenced text, so the allowlist keeps a
new node silent until somebody declares it safe.

### match

| Field | Type |
|---|---|
| `ingredient_name` | string |
| `status` | `matched`, `substituted`, `unavailable` |
| `spin_id` | string or null |
| `product_name` | string or null |
| `pack_size` | string or null |
| `unit_price` | number or null |
| `image_url` | string or null |
| `rating` | object with `value` and `count`, or null |
| `quantity` | integer or null |
| `substitution_reason` | string or null |

### error

A failure after the first byte cannot be a status code, because the response
has started. It arrives here instead, and the stream then closes cleanly.

| `code` | `action` | Meaning |
|---|---|---|
| `provider_auth` | `reconnect_swiggy` | The Swiggy token is dead |
| `provider_unavailable` | `retry` | Swiggy is unreachable |
| `provider_rejected` | `retry` | Swiggy refused the operation |
| `agent_failed` | `retry` | The agent could not finish the turn |

Anything that is not a known domain failure is a bug. It still raises.

## Persistence

| Item | Owner | Provisioned by |
|---|---|---|
| `chat_session`, `chat_message` | The app | `alembic upgrade head` |
| Checkpoint tables | LangGraph | `scripts/setup_checkpointer.py` |

The checkpointer runs DDL, so it is a deployment step. The application never
runs it at startup. Every process running DDL on every start is a wasted round
trip at best, and a migration race at worst.

The checkpoint serializer takes an explicit allowlist of types,
`_CHECKPOINTED_MSGPACK_TYPES` in `app/agent/graph.py`. Add a type there when a
node puts a new model into the state, or strict MsgPack rejects the resume.

## Tracing

LangGraph and LangChain trace to LangSmith when `LANGSMITH_TRACING`,
`LANGSMITH_API_KEY`, and `LANGSMITH_PROJECT` are real process environment
variables. `main.py` calls `load_dotenv()` before it imports any application
module, because the settings classes never export `.env` to the process
environment.

Run `uv run python scripts/agent_smoke.py` to invoke the graph once and
confirm that LangSmith recorded the run.
