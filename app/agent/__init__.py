"""The Cue LangGraph agent runtime.

This package is the standalone graph scaffold (CUE-21): a compiled StateGraph
with a swappable chat-model provider, LangSmith tracing, and a Postgres
checkpointer. Recipe generation, photo parsing, normalization, and substitution
nodes plug into `graph.build_graph()` in later issues without touching this
scaffold.
"""

from __future__ import annotations
