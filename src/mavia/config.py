"""Central configuration, loaded from environment / .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="MAVIA_",
        extra="ignore",
    )

    # --- LLM ---
    anthropic_api_key: SecretStr | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    llm_model: str = "claude-opus-5"
    llm_max_tokens: int = 2048

    # --- Vector memory ---
    qdrant_url: str | None = None
    qdrant_api_key: SecretStr | None = None
    qdrant_path: Path = REPO_ROOT / "artifacts" / "qdrant_storage"
    qdrant_collection: str = "defect_history"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Vision ---
    anomaly_model: str = "patchcore"
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"

    # --- Paths ---
    data_dir: Path = REPO_ROOT / "data"
    models_dir: Path = REPO_ROOT / "models"
    artifacts_dir: Path = REPO_ROOT / "artifacts"

    # --- Decision policy ---
    high_risk_threshold: float = 0.75
    retrieval_top_k: int = 3

    @property
    def mvtec_dir(self) -> Path:
        return self.data_dir / "mvtec_ad"

    @property
    def audit_log_path(self) -> Path:
        return self.artifacts_dir / "audit" / "audit_log.jsonl"

    @property
    def uses_qdrant_server(self) -> bool:
        return bool(self.qdrant_url)

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.models_dir, self.artifacts_dir):
            path.mkdir(parents=True, exist_ok=True)
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
