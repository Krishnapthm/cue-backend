# Add a graph node

This guide adds one node to the agent graph. It assumes that you already ran
the service once. Read [Getting started](../getting-started.md) first if you
have not.

Use the [agent graph reference](../reference/agent-graph.md) for the shapes
this guide refers to.

## Decide what the node is

Answer three questions before you write code:

1. **Does the node need a model?** A deterministic node is cheaper, faster,
   and testable. `normalize_ingredients`, `report_cart`, and `refuse` take no
   model at all.
2. **Does the node reach off-box?** If it does, it needs a retry policy, and
   it must be safe to repeat.
3. **Does the node write new values into the state?** If it does, the values
   must serialize, and their types must join the serializer allowlist.

## 1. Write the node

Create a module in `app/agent/nodes/`. A node takes the state, and takes the
runtime when it needs the request-scoped handles.

```python
"""One-line statement of what this node does."""

from __future__ import annotations

from typing import Any

from langgraph.runtime import Runtime

from app.agent.context import CueContext
from app.agent.state import AgentState


async def estimate_budget_node(
    state: AgentState, runtime: Runtime[CueContext]
) -> dict[str, Any]:
    """Estimate the cart total before the fan-out runs.

    Args:
        state: The current graph state.
        runtime: The turn's runtime context, which carries the database
            session and the user.

    Returns:
        A partial state update.
    """
    session = runtime.context.session
    ...
    return {"budget_estimate": estimate}
```

Rules that the existing nodes follow:

- Return a partial update, never the whole state.
- Read the database session from `runtime.context`, never from a global
  factory. The node's writes then join the request's unit of work.
- Put nothing in the state that fails to serialize.
- Call a domain service for business logic. A node never writes SQL, and never
  calls Swiggy directly.

## 2. Ask for a model by role, if you need one

A node never names a model.

```python
from app.agent.config import ModelRole
from app.agent.providers import get_chat_model

model = get_chat_model(ModelRole.ORDER_STATUS)
```

To add a role, edit `app/agent/config.py`:

1. Add the member to `ModelRole`, with a comment that states the cost and
   quality profile.
2. Add the `MODEL_<ROLE>` field to `AgentSettings`, with a default.
3. Add the `case` to `model_for`.
4. Set `reasoning_effort` only for a model that offers the dial. `gpt-4o-mini`
   rejects the argument with HTTP 400.

Then document the new variable in the
[configuration reference](../reference/configuration.md#agent).

## 3. Force the output into a schema

A model that returns free text becomes a parsing problem later. Declare a
Pydantic model in `app/agent/schemas.py`, and use structured output:

```python
decision = await model.with_structured_output(BudgetVerdict).ainvoke(prompt)
```

Use a closed enum for any label. A classifier that returns an unrecognized
value then fails validation, instead of being read as a valid answer.

## 4. Register the node

Edit `app/agent/graph.py`:

```python
ESTIMATE_BUDGET = "estimate_budget"

builder.add_node(ESTIMATE_BUDGET, estimate_budget_node)
builder.add_edge(NORMALIZE_INGREDIENTS, ESTIMATE_BUDGET)
builder.add_edge(ESTIMATE_BUDGET, CONFIRM_CHECKLIST)
```

Add `retry_policy=NETWORK_RETRY` when the node reaches off-box:

```python
builder.add_node(ESTIMATE_BUDGET, estimate_budget_node, retry_policy=NETWORK_RETRY)
```

Update the diagram in the `build_graph` docstring. It is the map that every
later reader trusts.

## 5. Extend the state, if the node needs a new field

Edit `app/agent/state.py`. Mark a new field `NotRequired`, so the state
literals that already exist stay valid:

```python
budget_estimate: NotRequired[Decimal | None]
```

Add a reducer when parallel branches write to the field:

```python
findings: Annotated[NotRequired[list[Finding]], operator.add]
```

Without the reducer, the last branch to finish overwrites every other result,
and the failure looks like a correct short answer.

Document the field in `AgentState`'s docstring, and in the
[agent graph reference](../reference/agent-graph.md#state).

## 6. Add your types to the serializer allowlist

Any Pydantic model or enum that lands in the state must join
`_CHECKPOINTED_MSGPACK_TYPES` in `app/agent/graph.py`:

```python
_CHECKPOINTED_MSGPACK_TYPES: tuple[type[Any], ...] = (
    ...,
    BudgetVerdict,
)
```

Skip this step, and the resume of a paused thread fails under strict MsgPack.

## 7. Decide whether the node may stream prose

A node's tokens reach the user only when the node is in `PROSE_NODES`:

```python
PROSE_NODES: frozenset[str] = frozenset({ORDER_STATUS})
```

Add a node here only when its model output is prose written for the user. A
node that runs under `with_structured_output` must stay out, because its
tokens are fragments of an internal schema, and one of those fields holds
attacker-influenced text.

## 8. Pause only where a pause is safe

Call `interrupt()` only before an operation that spends money or changes
user-visible state. Two rules hold today:

- Nothing inside the ingredient fan-out may pause. The user already gave the
  instruction, and those searches are spent on that instruction.
- No operation that is unsafe to repeat may sit before a pause. A pause ends
  the invocation, and the run replays from the checkpoint.

## 9. Write the tests

Tests live in `tests/agent/`, and they mirror the node modules. No test may
call a real model, so replace `get_chat_model` with a fake:

```python
def test_estimate_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _FakeModel(BudgetVerdict(total=Decimal("450")))
    monkeypatch.setattr(budget_node, "get_chat_model", lambda role: model)
```

`tests/conftest.py` removes `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`, so a
missed stub fails loudly rather than billing a live request.

Cover the failure path as well as the happy path. State which failures are
about one item, and which end the turn.

## 10. Run the checks

```bash
uv run ruff check --fix . && uv run ruff format .
uv run mypy .
uv run pytest
```

All three must pass before you commit. The test suite starts a real Postgres
container, so Docker must run.

## Checklist

- [ ] The node returns a partial update
- [ ] The node reads its session from `runtime.context`
- [ ] A model call asks for a role, not a model id
- [ ] A model call uses a schema
- [ ] The node is registered, and the edges are in place
- [ ] The `build_graph` diagram is up to date
- [ ] A new state field is `NotRequired`, and has a reducer when it needs one
- [ ] New state types are in the serializer allowlist
- [ ] `PROSE_NODES` is unchanged, unless the output is prose for the user
- [ ] Tests cover the happy path and the failure path
- [ ] Ruff, mypy, and pytest pass
