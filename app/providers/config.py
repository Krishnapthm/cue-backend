from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class ProviderSettings(BaseSettings):
    """Swiggy OAuth 2.1 + PKCE client settings."""

    model_config = SettingsConfigDict(
        env_prefix="SWIGGY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Optional: unset in local dev, where the Swiggy link flow is never
    # exercised. Routes/services that need one of these validate it lazily
    # and raise ProviderNotConfiguredError rather than failing app startup.
    CLIENT_ID: str | None = None
    # Our own callback URL, registered with Swiggy; must match exactly on
    # both the authorize and token-exchange calls.
    REDIRECT_URI: str | None = None
    # Fixed app deep link the callback redirects to once linking finishes,
    # success or failure, so the client can resume the pending action (R2.4).
    APP_CALLBACK_DEEP_LINK: str | None = None
    # Fernet key encrypting access_token_ct / code_verifier_ct at rest.
    TOKEN_ENCRYPTION_KEY: str | None = None


@lru_cache
def get_provider_settings() -> ProviderSettings:
    """Return the cached provider settings."""
    return ProviderSettings()


provider_settings = get_provider_settings()
