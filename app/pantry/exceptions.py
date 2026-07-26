from __future__ import annotations

from fastapi import status

from app.exceptions import AppError, NotFoundError


class PantryItemNotFoundError(NotFoundError):
    """The requested pantry item does not exist or is not owned by the caller.

    Both cases are the same 404 on purpose: a caller must not be able to
    probe for the existence of another user's items by status code.
    """

    detail = "Pantry item not found."


class PantryItemNameConflictError(AppError):
    """Renaming an item would collide with another of the caller's items.

    Only reachable from `PATCH`; `POST` resolves the same collision by
    updating the existing row instead of erroring.
    """

    status_code = status.HTTP_409_CONFLICT
    detail = "Another pantry item already uses that name."
