"""Pydantic models and DTOs used by the pipeline and HTTP API."""

from datetime import datetime, timezone
from enum import StrEnum
from typing import Literal
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


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


class AIRecommendation(StrEnum):
    """Allowed AI recommendations for a generated reel."""

    UPLOAD = "UPLOAD"
    REVIEW = "REVIEW"
    SKIP = "SKIP"


class AIReviewStatus(StrEnum):
    """Per-reel AI review lifecycle state."""

    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class HumanReviewStatus(StrEnum):
    """Manual review state required before upload."""

    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    NEEDS_EDIT = "NEEDS_EDIT"


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
    audio_path: str | None = Field(default=None, alias="audioPath")
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
    audio_path: Path
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


class ReelClipDTO(BaseModel):
    """One generated vertical reel candidate."""

    id: str
    path: Path
    start: float
    end: float
    duration: float
    width: int = 1080
    height: int = 1920
    score: float | None = None
    rank: int | None = None


class ReelGenerationResultDTO(BaseModel):
    """Result returned after generating vertical reel candidates."""

    clips: list[ReelClipDTO]
    total_generated: int = Field(alias="totalGenerated")
    clip_seconds: int = Field(alias="clipSeconds")

    model_config = {"populate_by_name": True}


class AIScoreBreakdown(BaseModel):
    """Strict score breakdown returned by the local review model."""

    hook: float = Field(ge=0, le=10)
    gameplay: float = Field(ge=0, le=10)
    excitement: float = Field(ge=0, le=10)
    visual_clarity: float = Field(alias="visualClarity", ge=0, le=10)
    context_independence: float = Field(alias="contextIndependence", ge=0, le=10)
    payoff: float = Field(ge=0, le=10)
    pacing: float = Field(ge=0, le=10)
    audio_clarity: float = Field(alias="audioClarity", ge=0, le=10)
    technical_quality: float = Field(alias="technicalQuality", ge=0, le=10)
    upload_potential: float = Field(alias="uploadPotential", ge=0, le=10)

    model_config = {"populate_by_name": True}


class AIReviewResult(BaseModel):
    """Validated local model review result for one reel candidate."""

    scores: AIScoreBreakdown
    recommendation: Literal["UPLOAD", "REVIEW", "SKIP"]
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(max_length=240)
    detected_moment: str | None = Field(default=None, alias="detectedMoment", max_length=120)
    suggested_title: str = Field(alias="suggestedTitle", max_length=80)
    suggested_caption: str = Field(alias="suggestedCaption", max_length=300)
    hashtags: list[str] = Field(default_factory=list, max_length=10)
    issues: list[str] = Field(default_factory=list)

    model_config = {"populate_by_name": True}

    @field_validator("hashtags")
    @classmethod
    def normalize_hashtags(cls, value: list[str]) -> list[str]:
        """Ensure hashtags are concise and consistently prefixed."""

        normalized = []
        for item in value:
            tag = item.strip()
            if not tag:
                continue
            normalized.append(tag if tag.startswith("#") else f"#{tag}")
        return normalized[:10]


class HumanReviewUpdateDTO(BaseModel):
    """Request body for manual reel review decisions."""

    status: HumanReviewStatus
    notes: str | None = Field(default=None, max_length=1000)
    edited_title: str | None = Field(default=None, alias="editedTitle", max_length=80)
    edited_caption: str | None = Field(default=None, alias="editedCaption", max_length=300)
    edited_hashtags: list[str] | None = Field(default=None, alias="editedHashtags", max_length=10)

    model_config = {"populate_by_name": True}
