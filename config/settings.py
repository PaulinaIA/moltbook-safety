"""Centralized application settings using pydantic-settings."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide configuration settings.

    Settings can be overridden via environment variables prefixed with MOLTBOOK_.
    Example: MOLTBOOK_RATE_LIMIT_SECONDS=2.0
    """

    model_config = SettingsConfigDict(
        env_prefix="MOLTBOOK_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Base paths
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent,
        description="Root directory of the project",
    )

    @property
    def data_dir(self) -> Path:
        """Data directory for all pipeline outputs."""
        return self.project_root / "data"

    @property
    def raw_dir(self) -> Path:
        """Directory for cached HTML files."""
        return self.data_dir / "raw"

    @property
    def silver_dir(self) -> Path:
        """Directory for cleaned parquet files."""
        return self.data_dir / "silver"

    @property
    def gold_dir(self) -> Path:
        """Directory for feature-engineered parquet files."""
        return self.data_dir / "gold"

    @property
    def models_dir(self) -> Path:
        """Directory for model artifacts."""
        return self.data_dir / "models"

    # Database settings
    db_type: Literal["sqlite", "postgres"] = Field(
        default="sqlite",
        description="Database backend to use",
    )
    db_path: str = Field(
        default="data/moltbook.db",
        description="SQLite database path (relative to project root)",
    )
    postgres_url: str = Field(
        default="postgresql://user:pass@localhost:5432/moltbook",
        description="PostgreSQL connection URL (used when db_type=postgres)",
    )

    # Scraping settings
    base_url: str = Field(
        default="https://www.moltbook.com",
        description="Base URL for moltbook.com",
    )
    rate_limit_seconds: float = Field(
        default=1.0,
        ge=0.5,
        description="Minimum seconds between requests",
    )
    max_retries: int = Field(
        default=3,
        ge=1,
        description="Maximum number of retry attempts",
    )
    retry_base_delay: float = Field(
        default=1.0,
        description="Base delay for exponential backoff (seconds)",
    )
    request_timeout: int = Field(
        default=30,
        description="Request timeout in seconds",
    )
    user_agent: str = Field(
        default="MoltbookScraper/1.0 (Academic Research Project)",
        description="User-Agent header for requests",
    )
    headless: bool = Field(
        default=True,
        description="Run browser in headless mode",
    )

    # Processing settings
    batch_size: int = Field(
        default=1000,
        description="Batch size for database operations",
    )

    # Model settings
    target_column: str = Field(
        default="karma",
        description="Target variable for regression",
    )
    max_models: int = Field(
        default=10,
        description="Maximum models for H2O AutoML",
    )
    max_runtime_secs: int = Field(
        default=300,
        description="Maximum runtime for H2O AutoML in seconds",
    )

    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        for directory in [
            self.data_dir,
            self.raw_dir,
            self.silver_dir,
            self.gold_dir,
            self.models_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def db_connection_string(self) -> str:
        """Get the appropriate database connection string."""
        if self.db_type == "sqlite":
            db_full_path = self.project_root / self.db_path
            db_full_path.parent.mkdir(parents=True, exist_ok=True)
            return f"sqlite:///{db_full_path}"
        return self.postgres_url


# Global settings instance
settings = Settings()
