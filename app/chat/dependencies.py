from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends

from app.agent.graph import CueGraph, open_compiled_graph


async def agent_graph() -> AsyncIterator[CueGraph]:
    """Yield the compiled agent graph for the duration of one request.

    The graph is opened per request, which also opens a checkpointer
    connection. If that proves too costly under load, the follow-up is an
    app-lifespan-scoped compiled graph - this dependency is the seam where
    that change would land, so no caller needs to know either way.

    Yields:
        The compiled, checkpointed graph.
    """
    async with open_compiled_graph() as graph:
        yield graph


AgentGraph = Annotated[CueGraph, Depends(agent_graph)]
