from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic import PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    """Deployment environment the app is running in."""

    LOCAL = "local"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """Global application settings, loaded from the environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    DATABASE_URL: PostgresDsn
    ENVIRONMENT: Environment = Environment.LOCAL

    # Browser origins allowed to call the API (Expo's web target in local dev).
    # Defaults to empty so an unconfigured deployment is closed rather than
    # open. Never a wildcard: the client sends a bearer token on every call,
    # so the origin list is the only thing narrowing who may do that from a
    # browser. Set as a JSON array, e.g.
    # CORS_ALLOW_ORIGINS='["http://localhost:8081"]'.
    CORS_ALLOW_ORIGINS: list[str] = []

    # Verify connections before handing them out of the pool. Supabase closes
    # idle connections server-side, so without this the first query after an
    # idle period fails.
    DATABASE_POOL_PRE_PING: bool = True
    DATABASE_POOL_SIZE: int = 5
    DATABASE_MAX_OVERFLOW: int = 10
    DATABASE_ECHO: bool = False


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings."""
    return Settings()


settings = get_settings()
