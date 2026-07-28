"""Provision the LangGraph checkpointer's tables (CUE-93).

A **deployment step**, run once against the database beside
`alembic upgrade head` - not something the application does when it boots.
`AsyncPostgresSaver.setup()` is DDL, and having every process issue it on
every start is a wasted round-trip per boot at best and a migration race at
worst.

    uv run python scripts/setup_checkpointer.py

Reads `DATABASE_URL` from the environment like the rest of the app, or takes
an explicit DSN as its first argument.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

# Run as a script rather than imported as a module, so the repository root is
# not already on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.graph import setup_checkpointer

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger("setup_checkpointer")


async def main() -> None:
    """Create the checkpoint tables if they are not already there."""
    dsn = sys.argv[1] if len(sys.argv) > 1 else None
    await setup_checkpointer(dsn)
    logger.info("Checkpointer tables are provisioned.")


if __name__ == "__main__":
    asyncio.run(main())
