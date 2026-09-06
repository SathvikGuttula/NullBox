from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "VoxShield"
    app_env: str = "development"
    debug: bool = True

    database_url: str = (
        "postgresql+asyncpg://voxshield:voxshield@localhost:5432/voxshield"
    )

    cors_origins: str = "http://localhost:3000"

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


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()