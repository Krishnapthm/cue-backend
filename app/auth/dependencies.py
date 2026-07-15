from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.auth import service
from app.auth.exceptions import InvalidTokenError
from app.auth.schemas import FirebaseClaims
from app.database import DbSession
from app.models.user import User


async def bearer_token(authorization: Annotated[str | None, Header()] = None) -> str:
    """Extract the raw JWT from the `Authorization: Bearer <token>` header."""
    if authorization is None:
        raise InvalidTokenError("Missing Authorization header.")

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise InvalidTokenError("Authorization header must be a Bearer token.")

    return token


async def verified_claims(
    token: Annotated[str, Depends(bearer_token)],
) -> FirebaseClaims:
    """Verify the bearer token and return its claims."""
    return await service.verify_firebase_id_token(token)


async def current_user(
    claims: Annotated[FirebaseClaims, Depends(verified_claims)],
    session: DbSession,
) -> User:
    """Resolve verified Firebase claims to the matching Cue user.

    Provisions the user row on first sign-in and refreshes email/display name
    on every call after, via a single upsert - never a select-then-insert,
    which would race under concurrent first requests for the same account.
    """
    stmt = (
        pg_insert(User)
        .values(
            firebase_uid=claims.sub,
            email=claims.email,
            display_name=claims.name,
        )
        .on_conflict_do_update(
            constraint="uq_user_firebase_uid",
            set_={"email": claims.email, "display_name": claims.name},
        )
        .returning(User)
    )
    result = await session.execute(stmt)
    user = result.scalar_one()
    await session.commit()
    return user


CurrentUser = Annotated[User, Depends(current_user)]
