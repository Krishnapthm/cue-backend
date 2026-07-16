"""Image loading seam for recipe-photo parsing (CUE-23).

`app.agent.nodes.recipe.parse_recipe_photo_node` needs raw image bytes for a
Supabase Storage object path, but nodes receive only `AgentState` - no DB
session, no HTTP client. This module is the provider-swap-shaped seam that
solves that, mirroring `app.agent.providers.get_chat_model`: a narrow
`ImageStore` protocol nodes depend on, a concrete `SupabaseImageStore`, and a
`get_image_store` factory. Tests monkeypatch `get_image_store` exactly like
`get_chat_model`, so the node stays unit-testable with no network access.
"""

from __future__ import annotations

import logging
from typing import Protocol

import httpx

from app.agent.config import AgentSettings, agent_settings

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT_SECONDS = 10.0


class ImageStore(Protocol):
    """Loads raw image bytes for a Supabase Storage object path.

    Graph nodes depend on this Protocol rather than `SupabaseImageStore`
    directly, so swapping the storage backend (or substituting a fake in
    tests) never touches `app.agent.nodes.recipe`.
    """

    async def load(self, object_path: str) -> bytes:
        """Return the raw bytes stored at `object_path`."""
        ...


class SupabaseImageStore:
    """Fetches recipe image bytes from Supabase Storage's public object URL.

    Public object URL format:
    `{SUPABASE_URL}/storage/v1/object/public/{bucket}/{object_path}`.
    """

    def __init__(
        self,
        settings: AgentSettings = agent_settings,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Initialize the store.

        Args:
            settings: Agent settings to read `SUPABASE_URL` and
                `RECIPE_IMAGE_BUCKET` from; defaults to the process-wide
                cached settings.
            transport: Optional `httpx` transport override. Left `None` in
                production (a real connection is opened per `load` call);
                tests inject an `httpx.MockTransport` here to assert the
                request without any real network access.
        """
        self._settings = settings
        self._transport = transport

    async def load(self, object_path: str) -> bytes:
        """Fetch and return the raw bytes for `object_path`.

        Args:
            object_path: The Supabase Storage object path, as stored on
                `ChatMessage.payload` for `kind='image'` (see
                `app/models/chat.py`).

        Returns:
            The raw image bytes.

        Raises:
            RuntimeError: `AGENT_SUPABASE_URL` is not configured, or the
                fetch failed (network error or a non-200 response).
        """
        if not self._settings.SUPABASE_URL:
            raise RuntimeError(
                "Cannot load recipe image: AGENT_SUPABASE_URL is not configured."
            )
        url = (
            f"{self._settings.SUPABASE_URL}/storage/v1/object/public/"
            f"{self._settings.RECIPE_IMAGE_BUCKET}/{object_path}"
        )
        try:
            async with httpx.AsyncClient(
                timeout=_REQUEST_TIMEOUT_SECONDS, transport=self._transport
            ) as client:
                response = await client.get(url)
        except httpx.HTTPError as exc:
            logger.warning("Failed to fetch recipe image %r: %s", object_path, exc)
            raise RuntimeError(f"Failed to fetch recipe image {object_path!r}") from exc

        if response.status_code != httpx.codes.OK:
            logger.warning(
                "Recipe image fetch for %r returned HTTP %s",
                object_path,
                response.status_code,
            )
            raise RuntimeError(
                f"Failed to fetch recipe image {object_path!r}: "
                f"HTTP {response.status_code}"
            )
        return response.content


def get_image_store(settings: AgentSettings = agent_settings) -> ImageStore:
    """Return the configured image store.

    Analogous to `app.agent.providers.get_chat_model`: nodes depend on the
    `ImageStore` protocol returned here, never on `SupabaseImageStore`
    directly, so the storage backend is swappable with no edit to any node.

    Args:
        settings: Agent settings to read storage configuration from;
            defaults to the process-wide cached settings.

    Returns:
        A ready-to-use `ImageStore`.
    """
    return SupabaseImageStore(settings)
