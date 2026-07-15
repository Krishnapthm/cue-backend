from __future__ import annotations

from pydantic import BaseModel, Field


class FirebaseClaims(BaseModel):
    """Verified claims taken from a Firebase ID token.

    `sub` is Firebase's stable per-user identifier; it is what
    `app.models.User.firebase_uid` is keyed on.
    """

    sub: str = Field(min_length=1)
    email: str
    email_verified: bool = False
    name: str | None = None
