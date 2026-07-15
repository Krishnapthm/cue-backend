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

    CLIENT_ID: str
    # Our own callback URL, registered with Swiggy; must match exactly on
    # both the authorize and token-exchange calls.
    REDIRECT_URI: str
    # Fixed app deep link the callback redirects to once linking finishes,
    # success or failure, so the client can resume the pending action (R2.4).
    APP_CALLBACK_DEEP_LINK: str
    # Fernet key encrypting access_token_ct / code_verifier_ct at rest.
    TOKEN_ENCRYPTION_KEY: str


@lru_cache
def get_provider_settings() -> ProviderSettings:
    """Return the cached provider settings."""
    return ProviderSettings()


provider_settings = get_provider_settings()
