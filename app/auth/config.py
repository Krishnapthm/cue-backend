from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class AuthSettings(BaseSettings):
    """Firebase Auth verification settings."""

    model_config = SettingsConfigDict(
        env_prefix="AUTH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    FIREBASE_PROJECT_ID: str

    @property
    def firebase_issuer(self) -> str:
        """The `iss` claim every valid Firebase ID token for this project carries."""
        return f"https://securetoken.google.com/{self.FIREBASE_PROJECT_ID}"


@lru_cache
def get_auth_settings() -> AuthSettings:
    """Return the cached auth settings."""
    return AuthSettings()


auth_settings = get_auth_settings()
