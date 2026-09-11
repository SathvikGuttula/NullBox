from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "VoxShield"
    app_env: str = "development"
    debug: bool = True

    database_url: str = (
        "postgresql+asyncpg://voxshield:voxshield@localhost:5432/voxshield"
    )

    # Comma-separated exact origins allowed to call the API from a browser.
    # The Next.js dev server is the default; add a deployed frontend here.
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    # Optional regex, for hosts whose origin is not known ahead of time.
    #
    # Vercel gives every preview deployment its own hostname
    # (voxshield-<hash>-<scope>.vercel.app), so an exact allowlist breaks on
    # every push. Set this to something like
    #     ^https://[a-z0-9-]+\.vercel\.app$
    # to cover them. Left empty by default: matching more origins than you
    # meant to is exactly the mistake CORS exists to prevent, and this API has
    # no authentication in front of it.
    cors_origin_regex: str = ""

    risk_update_interval_ms: int = 500

    audio_sample_rate: int = 16000
    audio_channels: int = 1
    audio_chunk_seconds: int = 2

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def cors_origin_list(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.cors_origins.split(",")
            if origin.strip()
        ]

    @property
    def cors_origin_pattern(self) -> str | None:
        return self.cors_origin_regex.strip() or None


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()