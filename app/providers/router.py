from __future__ import annotations

from fastapi import APIRouter, status
from fastapi.responses import RedirectResponse

from app.auth.dependencies import CurrentUser
from app.database import DbSession
from app.providers import service
from app.providers.config import provider_settings
from app.providers.exceptions import InvalidOAuthStateError, SwiggyTokenExchangeError
from app.providers.schemas import AuthorizeResponse

router = APIRouter(prefix="/providers/swiggy", tags=["providers"])


@router.post(
    "/authorize",
    response_model=AuthorizeResponse,
    status_code=status.HTTP_200_OK,
    summary="Start the Swiggy OAuth 2.1 + PKCE link flow",
)
async def authorize(user: CurrentUser, session: DbSession) -> AuthorizeResponse:
    """Return the Swiggy consent URL for the signed-in Cue user to open."""
    return await service.create_authorization(session, user)


@router.get(
    "/callback",
    summary="Swiggy OAuth redirect target",
    responses={
        status.HTTP_307_TEMPORARY_REDIRECT: {
            "description": "Redirects back into the Cue app"
        },
    },
)
async def callback(
    session: DbSession,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
) -> RedirectResponse:
    """Exchange the Swiggy code for a token, then hand control back to the app.

    Always redirects to the app's fixed deep link, on success or failure, so
    the mobile action queue can resume the pending action (R2.4) either way.
    """
    if error or not code or not state:
        return RedirectResponse(
            f"{provider_settings.APP_CALLBACK_DEEP_LINK}?swiggy_link=error"
        )

    try:
        await service.complete_authorization(session, state=state, code=code)
    except (InvalidOAuthStateError, SwiggyTokenExchangeError):
        return RedirectResponse(
            f"{provider_settings.APP_CALLBACK_DEEP_LINK}?swiggy_link=error"
        )

    return RedirectResponse(
        f"{provider_settings.APP_CALLBACK_DEEP_LINK}?swiggy_link=success"
    )
