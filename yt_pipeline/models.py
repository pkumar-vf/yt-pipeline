"""Pydantic models and DTOs used by the pipeline and HTTP API."""

from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc)


class VideoStatus(StrEnum):
    """Pipeline state for a video document."""

    DISCOVERED = "DISCOVERED"
    DOWNLOADING = "DOWNLOADING"
    DOWNLOADED = "DOWNLOADED"
    TRANSCRIBED = "TRANSCRIBED"
    REELS_GENERATED = "REELS_GENERATED"
    AI_REVIEWED = "AI_REVIEWED"
    READY_FOR_UPLOAD = "READY_FOR_UPLOAD"
    UPLOADED = "UPLOADED"
    FAILED = "FAILED"


class StageName(StrEnum):
    """Known processing stage keys stored on each video."""

    DOWNLOAD = "download"
    TRANSCRIPTION = "transcription"
    REELS = "reels"
    AI_REVIEW = "aiReview"
    INSTAGRAM = "instagram"


class StageState(BaseModel):
    """Persisted result and progress information for one pipeline stage."""

    completed: bool = False
    timestamp: datetime | None = None
    path: str | None = None
    error: str | None = None
    transcript_path: str | None = Field(default=None, alias="transcriptPath")
    subtitle_path: str | None = Field(default=None, alias="subtitlePath")
    language: str | None = None
    total_segments: int | None = Field(default=None, alias="totalSegments")
    model: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


class VideoDocument(BaseModel):
    """MongoDB representation for one YouTube video and its stage state."""

    video_id: str = Field(alias="videoId")
    title: str
    published_at: datetime | None = Field(default=None, alias="publishedAt")
    status: VideoStatus = VideoStatus.DISCOVERED
    local_path: str | None = Field(default=None, alias="localPath")
    created_at: datetime = Field(default_factory=utc_now, alias="createdAt")
    updated_at: datetime = Field(default_factory=utc_now, alias="updatedAt")
    stages: dict[str, StageState] = Field(
        default_factory=lambda: {stage.value: StageState() for stage in StageName}
    )

    model_config = {"populate_by_name": True, "use_enum_values": True}

    def to_mongo(self) -> dict[str, Any]:
        """Convert this model into a MongoDB-friendly dictionary."""

        return self.model_dump(by_alias=True, mode="python")


class VideoSummaryDTO(BaseModel):
    """Compact video shape returned to the stage overview UI."""

    video_id: str
    title: str
    status: VideoStatus
    updated_at: datetime
    stages: dict[str, StageState]


class DiscoveredVideoDTO(BaseModel):
    """Video metadata returned by the downloader before persistence."""

    video_id: str
    title: str
    webpage_url: str
    published_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DownloadedVideoDTO(BaseModel):
    """Video metadata returned after a successful download."""

    video_id: str
    title: str
    local_path: Path
    metadata: dict[str, Any] = Field(default_factory=dict)


class TranscriptWord(BaseModel):
    """One word-level transcription result."""

    word: str
    start: float
    end: float
    probability: float | None = None


class TranscriptSegment(BaseModel):
    """One segment-level transcription result with optional word timings."""

    id: int
    start: float
    end: float
    text: str
    words: list[TranscriptWord] = Field(default_factory=list)


class TranscriptMetadata(BaseModel):
    """Transcript document metadata written before streamed segments."""

    language: str
    duration: float


class TranscriptionResultDTO(BaseModel):
    """Persistence result returned by the transcription export step."""

    transcript_path: Path
    subtitle_path: Path
    language: str
    duration: float
    total_segments: int
    model: str
