"""Application configuration loaded from environment variables."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings shared by the scheduler, pipeline, and UI."""

    channel_url: str = Field(default="", alias="YT_CHANNEL_URL")
    mongo_uri: str = Field(default="mongodb://localhost:27017", alias="MONGO_URI")
    mongo_db: str = Field(default="yt_pipeline", alias="MONGO_DB")
    videos_collection: str = Field(default="videos", alias="MONGO_VIDEOS_COLLECTION")
    downloads_dir: Path = Field(default=Path("downloads"), alias="DOWNLOADS_DIR")
    reels_dir: Path = Field(default=Path("reels"), alias="REELS_DIR")
    transcripts_dir: Path = Field(default=Path("transcripts"), alias="TRANSCRIPTS_DIR")
    review_assets_dir: Path = Field(default=Path("review-assets"), alias="REVIEW_ASSETS_DIR")
    reel_clip_seconds: int = Field(default=30, alias="REEL_CLIP_SECONDS")
    reel_max_clips: int = Field(default=20, alias="REEL_MAX_CLIPS")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="qwen3-vl:8b", alias="OLLAMA_MODEL")
    ollama_timeout_seconds: int = Field(default=180, alias="OLLAMA_TIMEOUT_SECONDS")
    ollama_max_retries: int = Field(default=2, alias="OLLAMA_MAX_RETRIES")
    ai_review_enabled: bool = Field(default=True, alias="AI_REVIEW_ENABLED")
    ai_review_concurrency: int = Field(default=1, alias="AI_REVIEW_CONCURRENCY")
    ai_review_frame_count: int = Field(default=8, alias="AI_REVIEW_FRAME_COUNT")
    ai_review_transcript_context_seconds: int = Field(default=5, alias="AI_REVIEW_TRANSCRIPT_CONTEXT_SECONDS")
    ai_review_max_overlap_ratio: float = Field(default=0.60, alias="AI_REVIEW_MAX_OVERLAP_RATIO")
    ai_review_duplicate_penalty: float = Field(default=1.5, alias="AI_REVIEW_DUPLICATE_PENALTY")
    review_proxy_max_width: int = Field(default=480, alias="REVIEW_PROXY_MAX_WIDTH")
    review_proxy_fps: int = Field(default=8, alias="REVIEW_PROXY_FPS")
    review_proxy_crf: int = Field(default=30, alias="REVIEW_PROXY_CRF")
    review_proxy_audio_bitrate: str = Field(default="48k", alias="REVIEW_PROXY_AUDIO_BITRATE")
    scheduler_minutes: int = Field(default=60, alias="SCHEDULER_MINUTES")
    ui_host: str = Field(default="127.0.0.1", alias="UI_HOST")
    ui_port: int = Field(default=8000, alias="UI_PORT")

    model_config = SettingsConfigDict(env_file=".env", populate_by_name=True)

    def ensure_dirs(self) -> None:
        """Create directories that pipeline stages write files into."""

        self.downloads_dir.mkdir(parents=True, exist_ok=True)
        self.reels_dir.mkdir(parents=True, exist_ok=True)
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        self.review_assets_dir.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    """Return a validated settings object for the current process."""

    return Settings()
