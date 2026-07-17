from __future__ import annotations

from fastapi import status

from app.exceptions import AppError


class RecipeGenerationError(AppError):
    """The model could not produce a valid structured recipe.

    Raised after the single retry (see `app.agent.nodes.recipe`) still fails
    Pydantic validation against `GeneratedRecipe`, so callers never see a raw
    parsing/validation exception surface as an unhandled 500.
    """

    status_code = status.HTTP_502_BAD_GATEWAY
    detail = "Failed to generate a structured recipe."
