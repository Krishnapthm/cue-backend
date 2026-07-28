from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.agent.graph import CueGraph


def agent_graph(request: Request) -> CueGraph:
    """Return the process-wide compiled agent graph.

    Cheap by design: the graph is built once in the FastAPI lifespan (see
    `app.main.lifespan`) and this hands out that same instance. It is shared
    across concurrent requests on purpose - nothing request-scoped is captured
    at compile time, because everything per-run travels in the run config
    (`thread_id`) or in `CueContext`.

    Args:
        request: The incoming request, for its app state.

    Returns:
        The compiled, checkpointed graph.

    Raises:
        RuntimeError: The app was started without its lifespan, so no graph was
            ever compiled. Raised loudly rather than compiling one here: a
            per-request graph is the exact thing CUE-93 removed, and silently
            reintroducing it under load would be worse than failing.
    """
    graph: CueGraph | None = getattr(request.app.state, "agent_graph", None)
    if graph is None:
        raise RuntimeError(
            "No compiled agent graph on app.state: the application was started "
            "without running its lifespan."
        )
    return graph


AgentGraph = Annotated[CueGraph, Depends(agent_graph)]
