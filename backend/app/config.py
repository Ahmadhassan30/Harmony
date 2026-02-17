"""Application configuration using pydantic-settings."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="HARMONY_",
        case_sensitive=False,
    )

    # Server
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = False

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Audio Processing
    sample_rate: int = 22050
    max_audio_duration_seconds: int = 600  # 10 minutes
    supported_formats: list[str] = ["mp3", "wav", "flac", "ogg", "m4a", "aac", "wma", "aiff"]

    # ML Models
    models_dir: Path = Path(__file__).parent / "ml" / "models"
    onnx_device: str = "auto"  # auto | cpu | cuda | coreml | directml

    # Cache
    cache_dir: Path = Path.home() / ".harmony" / "cache"
    cache_max_size_gb: float = 5.0

    # Processing
    max_workers: int = 4
    enable_separation: bool = True
    enable_extended_analysis: bool = True

    # Ensemble
    bpm_confidence_threshold: float = 0.7
    key_confidence_threshold: float = 0.6
    separation_quality_threshold: float = 0.85


settings = Settings()
