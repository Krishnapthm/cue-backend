"""Entrypoint shim so `uvicorn main:app` resolves to the real app in `app.main`."""

from __future__ import annotations

from dotenv import load_dotenv

# Must run before any app module is imported: our settings classes only parse
# .env into their own typed fields (pydantic-settings), they never export it
# to the real process environment. LangSmith, OpenAI, and Anthropic all read
# their env vars directly via os.getenv, so without this they never see a
# .env-only value - only a real exported env var. Doesn't override an
# already-set real env var (e.g. in prod), matching pydantic's own precedence.
load_dotenv()

from app.main import app  # noqa: E402

__all__ = ["app"]
