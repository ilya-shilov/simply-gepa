"""LLM connection settings."""

from typing import Optional

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """LLM connection settings from environment or direct initialization."""

    api_key: str = Field(
        default="",
        validation_alias=AliasChoices("GEPA_API_KEY", "OPENAI_API_KEY", "API_KEY", "api_key"),
    )
    model: str = "default"
    temperature: float = 0.7
    base_url: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


def get_settings() -> Settings:
    """Load settings from environment."""
    return Settings()
