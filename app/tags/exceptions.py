from __future__ import annotations

from app.exceptions import NotFoundError


class TagBindingNotFoundError(NotFoundError):
    """No binding for that tag UID, or it belongs to another user.

    Both cases answer the same 404 on purpose: a caller must not be able to
    discover that a tag UID is bound by someone else from the status code.
    """

    detail = "Tag binding not found."
