"""Per-invocation runtime context for the agent graph (CUE-85).

`AgentState` is checkpointed to Postgres, so nothing that fails to serialize -
an `AsyncSession` above all - may ever be put in it. Everything a node needs
that is *request-scoped rather than turn-scoped* travels here instead, reaching
nodes through `Runtime[CueContext]`:

```python
def compose_cart_node(state: AgentState, runtime: Runtime[CueContext]) -> dict:
    result = await cart_service.compose_cart(runtime.context.session, ...)
```

The context is deliberately **not** persisted. An `interrupt()` ends the
invocation, and the resume arrives on a *new* HTTP request holding a *new*
`AsyncSession`; a context supplied per-invocation is correct for free, whereas
a session smuggled into state would be stale (or unserializable) by then.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True)
class CueContext:
    """Request-scoped handles the graph's nodes need but must not checkpoint.

    Attributes:
        session: The request's active database session. Nodes call domain
            services with it, so their writes join the same unit of work as
            the HTTP request that triggered them - the reason this is a
            supplied session rather than a `get_session()` factory seam.
        user_id: The Cue user the turn belongs to.
        chat_session_id: The chat session being run; `str()` of it is also the
            checkpointer's `thread_id` (see `app/models/chat.py`).
        address_id: The Swiggy `addressId` the cart binds to. Resolved by
            `chat.service.run_turn` from `ChatSession.selected_address_id`
            before the graph is invoked - address selection is a precondition
            of a turn, never a decision the agent makes.
    """

    session: AsyncSession
    user_id: int
    chat_session_id: uuid.UUID
    address_id: str
