from __future__ import annotations

from fastapi import status

from app.exceptions import AppError


class InvalidOAuthStateError(AppError):
    """The OAuth state parameter is missing, unknown, expired, or already used."""

    status_code = status.HTTP_400_BAD_REQUEST
    detail = "Invalid or expired authorization state."


class SwiggyTokenExchangeError(AppError):
    """Swiggy's token endpoint rejected or failed the authorization code exchange."""

    status_code = status.HTTP_502_BAD_GATEWAY
    detail = "Failed to exchange authorization code with Swiggy."


class ProviderNotConfiguredError(AppError):
    """The Swiggy OAuth client is not configured on this deployment."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    detail = "Swiggy integration is not configured on this server."
