from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class FirebaseClaims(BaseModel):
    """Verified claims taken from a Firebase ID token.

    `sub` is Firebase's stable per-user identifier; it is what
    `app.models.User.firebase_uid` is keyed on.
    """

    sub: str = Field(min_length=1)
    email: str
    email_verified: bool = False
    name: str | None = None


class UserResponse(BaseModel):
    """The signed-in Cue user, as returned by `GET /auth/me`.

    `firebase_uid` is deliberately omitted: the client already holds it, and
    it is the identity this service authenticates on - there is no reason to
    echo it back over the wire.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    display_name: str | None
