"""The resolution path must stay deterministic (CUE-74).

Agent-freeness is load-bearing, not an implementation detail: a scan is a
physical rhythm, and routing it through the chat graph would put an LLM call
and its latency in the path of a gesture that has to feel instant. It is also
the kind of property that regresses silently - one convenience import of
`app.agent` and the whole endpoint is quietly on the model's clock.

So it is asserted structurally, over the import closure of `app.tags`, rather
than by observing one lucky request.
"""

from __future__ import annotations

import ast
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2] / "app"

ENTRY_MODULES = (
    "app.tags.router",
    "app.tags.service",
    "app.tags.dependencies",
)

# Nothing on the resolution path may reach the agent runtime, a graph, or a
# model client - directly or through any module it imports.
FORBIDDEN_PREFIXES = (
    "app.agent",
    "langgraph",
    "langchain",
    "anthropic",
    "openai",
)


def _module_path(module: str) -> Path | None:
    """Locate an `app.*` module's source file, if it is one of ours."""
    if not module.startswith("app."):
        return None
    relative = Path(*module.split(".")[1:])
    for candidate in (
        APP_ROOT / relative.with_suffix(".py"),
        APP_ROOT / relative / "__init__.py",
    ):
        if candidate.is_file():
            return candidate
    return None


def _imports_of(path: Path) -> set[str]:
    """Every module name imported by one source file."""
    tree = ast.parse(path.read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module)
            # `from app.pantry import service` names a module, not an object.
            imported.update(f"{node.module}.{alias.name}" for alias in node.names)
    return imported


def _import_closure(entry_modules: tuple[str, ...]) -> set[str]:
    """Every module reachable from `entry_modules`, following `app.*` edges."""
    seen: set[str] = set()
    queue = list(entry_modules)
    while queue:
        module = queue.pop()
        if module in seen:
            continue
        seen.add(module)
        path = _module_path(module)
        if path is None:
            continue
        queue.extend(_imports_of(path))
    return seen


def test_no_agent_graph_or_model_is_reachable_from_tag_resolution() -> None:
    closure = _import_closure(ENTRY_MODULES)

    offenders = sorted(
        module for module in closure if module.startswith(FORBIDDEN_PREFIXES)
    )

    assert offenders == []
    # Guard the guard: if the traversal stopped finding our own modules, the
    # assertion above would pass vacuously.
    assert "app.instamart.service" in closure
    assert "app.matching.substitution" in closure
