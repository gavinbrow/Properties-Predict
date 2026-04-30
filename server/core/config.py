"""Prod-local configuration.

All paths default to artifacts bundled inside the Prod/ folder so the
deployable is self-contained. Overrides can still come from environment
variables.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROD_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROD_ROOT / "data"
SQLITE_PREFIX = "sqlite:///"


def _resolve_local_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROD_ROOT / path
    return path.resolve()


def _resolve_database_url(value: str) -> str:
    if not isinstance(value, str) or not value.startswith(SQLITE_PREFIX):
        return value
    db_path = Path(value.removeprefix(SQLITE_PREFIX)).expanduser()
    if not db_path.is_absolute():
        db_path = PROD_ROOT / db_path
    return f"{SQLITE_PREFIX}{db_path.resolve()}"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROD_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: str = "prod"
    log_level: str = "INFO"
    host: str = "127.0.0.1"
    port: int = 8000

    gnn_models_root: Path = Field(default=DATA_ROOT / "trained_models" / "gnn")
    knn_models_root: Path = Field(default=DATA_ROOT / "trained_models" / "knn")
    database_url: str = f"sqlite:///{DATA_ROOT / 'lookup' / 'compounds.db'}"

    max_smiles_length: int = 500
    max_batch_size: int = 200
    engine_timeout_seconds: int = 60
    request_timeout_seconds: int = 120

    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    @field_validator("gnn_models_root", "knn_models_root", mode="before")
    @classmethod
    def _resolve_paths(cls, v):
        return _resolve_local_path(v)

    @field_validator("database_url", mode="before")
    @classmethod
    def _resolve_sqlite_url(cls, v):
        return _resolve_database_url(v)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_cors(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
