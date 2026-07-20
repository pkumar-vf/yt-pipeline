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
    scheduler_minutes: int = Field(default=60, alias="SCHEDULER_MINUTES")
    ui_host: str = Field(default="127.0.0.1", alias="UI_HOST")
    ui_port: int = Field(default=8000, alias="UI_PORT")

    model_config = SettingsConfigDict(env_file=".env", populate_by_name=True)

    def ensure_dirs(self) -> None:
        """Create directories that pipeline stages write files into."""

        self.downloads_dir.mkdir(parents=True, exist_ok=True)
        self.reels_dir.mkdir(parents=True, exist_ok=True)
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    """Return a validated settings object for the current process."""

    return Settings()
