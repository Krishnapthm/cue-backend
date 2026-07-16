"""Tracing config bridges settings onto the LangSmith env vars, and never
turns a missing key into a hard failure."""

from __future__ import annotations

import os

import pytest

from app.agent.config import AgentSettings
from app.agent.observability import configure_tracing


def _settings(*, tracing: bool, project: str = "cue-agent") -> AgentSettings:
    return AgentSettings(
        MODEL_NAME="claude-test-model",
        LANGSMITH_TRACING=tracing,
        LANGSMITH_PROJECT=project,
    )


def test_tracing_enabled_when_key_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test-key")
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)

    enabled = configure_tracing(_settings(tracing=True, project="cue-agent-test"))

    assert enabled is True
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_PROJECT"] == "cue-agent-test"


def test_tracing_degrades_to_warning_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    enabled = configure_tracing(_settings(tracing=True))

    # Observability is not a hard dependency: absence of a key disables tracing
    # rather than raising, so the graph can still run.
    assert enabled is False
    assert os.environ["LANGSMITH_TRACING"] == "false"


def test_tracing_disabled_by_config_even_with_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test-key")

    enabled = configure_tracing(_settings(tracing=False))

    assert enabled is False
    assert os.environ["LANGSMITH_TRACING"] == "false"
