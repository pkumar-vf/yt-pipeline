"""AI transcription stage powered by faster-whisper."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

from yt_pipeline.database import VideoRepository
from yt_pipeline.models import (
    StageName,
    TranscriptMetadata,
    TranscriptSegment,
    TranscriptionResultDTO,
    TranscriptWord,
    VideoDocument,
    VideoStatus,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL_NAME = "large-v3"


class WhisperLike(Protocol):
    """Protocol for the faster-whisper model methods used by this stage."""

    def transcribe(self, audio: str, **kwargs: Any) -> tuple[Iterable[Any], Any]:
        """Transcribe audio or video input and return segments plus metadata."""


class WhisperModelProvider:
    """Lazy singleton provider for the faster-whisper model."""

    _model: WhisperLike | None = None
    _model_name: str | None = None
    _lock = Lock()

    @classmethod
    def get(cls, model_name: str = DEFAULT_MODEL_NAME) -> WhisperLike:
        """Return a singleton Whisper model, loading it on first use."""

        with cls._lock:
            if cls._model is None or cls._model_name != model_name:
                from faster_whisper import WhisperModel

                cls._model = WhisperModel(model_name, device="auto", compute_type="int8")
                cls._model_name = model_name
            return cls._model


class SrtExporter:
    """Utility that writes segment timestamps and text in SRT format."""

    def write_segment(self, handle: Any, index: int, segment: TranscriptSegment) -> None:
        """Write one transcript segment as an SRT cue."""

        handle.write(f"{index}\n")
        handle.write(f"{format_srt_timestamp(segment.start)} --> {format_srt_timestamp(segment.end)}\n")
        handle.write(f"{segment.text.strip()}\n\n")


class TranscriptJsonStreamer:
    """Streaming JSON writer for faster-whisper transcript segments."""

    def __init__(self, path: Path) -> None:
        """Create a writer for one transcript JSON file."""

        self.path = path

    def write(
        self,
        *,
        metadata: TranscriptMetadata,
        segments: Iterable[Any],
        srt_path: Path,
        srt_exporter: SrtExporter,
        progress_callback: Callable[[int], None],
    ) -> int:
        """Stream transcript JSON and SRT files, returning the segment count."""

        total_segments = 0
        with self.path.open("w", encoding="utf-8") as json_file, srt_path.open("w", encoding="utf-8") as srt_file:
            json_file.write("{\n")
            json_file.write(f'  "language": {json.dumps(metadata.language)},\n')
            json_file.write(f'  "duration": {metadata.duration},\n')
            json_file.write('  "segments": [\n')

            for raw_segment in segments:
                segment = to_transcript_segment(raw_segment)
                if total_segments:
                    json_file.write(",\n")
                json_file.write("    ")
                json_file.write(segment.model_dump_json())
                total_segments += 1
                srt_exporter.write_segment(srt_file, total_segments, segment)
                progress_callback(total_segments)

            json_file.write("\n  ]\n")
            json_file.write("}\n")
        return total_segments


class TranscriptionStage:
    """Pipeline stage that transcribes downloaded videos and updates MongoDB."""

    def __init__(
        self,
        repo: VideoRepository,
        transcripts_dir: Path,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        model_provider: type[WhisperModelProvider] = WhisperModelProvider,
        logger: logging.Logger = LOGGER,
    ) -> None:
        """Create a transcription stage with explicit persistence and model dependencies."""

        self.repo = repo
        self.transcripts_dir = transcripts_dir
        self.model_name = model_name
        self.model_provider = model_provider
        self.logger = logger

    def process(self, video_id: str) -> None:
        """Transcribe one video by id and persist transcript paths in MongoDB."""

        started = time.monotonic()
        self.logger.info("Started transcription for video_id=%s", video_id)
        try:
            video = self._load_and_validate(video_id)
            if video is None:
                return

            result = self._transcribe(video)
            self.repo.complete_transcription(
                video_id,
                transcript_path=str(result.transcript_path),
                subtitle_path=str(result.subtitle_path),
                language=result.language,
                total_segments=result.total_segments,
                model=result.model,
            )
            self.logger.info("Mongo updated for video_id=%s", video_id)
            self.logger.info("Elapsed time %.2fs for video_id=%s", time.monotonic() - started, video_id)
        except Exception as exc:
            self.logger.exception("Transcription failed for video_id=%s", video_id)
            self.repo.fail_transcription(video_id, str(exc))

    def _load_and_validate(self, video_id: str) -> VideoDocument | None:
        """Load a video and validate that it is ready for transcription."""

        video = self.repo.get(video_id)
        if video is None:
            self.logger.error("Video not found for video_id=%s", video_id)
            return None

        stage = video.stages.get(StageName.TRANSCRIPTION.value)
        if stage and stage.completed:
            self.logger.info("Transcription already completed for video_id=%s", video_id)
            return None

        download = video.stages.get(StageName.DOWNLOAD.value)
        if not download or not download.completed:
            raise ValueError("Download stage must be completed before transcription.")
        if not video.local_path:
            raise ValueError("Video document is missing localPath.")

        local_path = Path(video.local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Local video path does not exist: {local_path}")
        return video

    def _transcribe(self, video: VideoDocument) -> TranscriptionResultDTO:
        """Run faster-whisper and export JSON plus SRT transcript artifacts."""

        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        transcript_path = self.transcripts_dir / f"{video.video_id}.json"
        subtitle_path = self.transcripts_dir / f"{video.video_id}.srt"

        model = self.model_provider.get(self.model_name)
        segments, info = model.transcribe(
            video.local_path or "",
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
        )

        language = str(getattr(info, "language", "unknown") or "unknown")
        duration = float(getattr(info, "duration", 0.0) or 0.0)
        self.logger.info("Detected language %s for video_id=%s", language, video.video_id)

        writer = TranscriptJsonStreamer(transcript_path)
        total_segments = writer.write(
            metadata=TranscriptMetadata(language=language, duration=duration),
            segments=segments,
            srt_path=subtitle_path,
            srt_exporter=SrtExporter(),
            progress_callback=self._log_progress,
        )
        self.logger.info("Saved transcript %s", transcript_path)

        return TranscriptionResultDTO(
            transcript_path=transcript_path,
            subtitle_path=subtitle_path,
            language=language,
            duration=duration,
            total_segments=total_segments,
            model=self.model_name,
        )

    def _log_progress(self, total_segments: int) -> None:
        """Log transcription progress every 100 processed segments."""

        if total_segments % 100 == 0:
            self.logger.info("Processed %s transcription segments", total_segments)


def to_transcript_segment(raw_segment: Any) -> TranscriptSegment:
    """Convert a faster-whisper segment object into a validated transcript model."""

    words = [
        TranscriptWord(
            word=str(_read_attr(word, "word", "")),
            start=float(_read_attr(word, "start", 0.0) or 0.0),
            end=float(_read_attr(word, "end", 0.0) or 0.0),
            probability=_optional_float(_read_attr(word, "probability", None)),
        )
        for word in (_read_attr(raw_segment, "words", None) or [])
    ]
    return TranscriptSegment(
        id=int(_read_attr(raw_segment, "id", 0) or 0),
        start=float(_read_attr(raw_segment, "start", 0.0) or 0.0),
        end=float(_read_attr(raw_segment, "end", 0.0) or 0.0),
        text=str(_read_attr(raw_segment, "text", "")),
        words=words,
    )


def format_srt_timestamp(seconds: float) -> str:
    """Format a float timestamp as an SRT timestamp."""

    milliseconds = round(max(seconds, 0.0) * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1_000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def _read_attr(value: Any, name: str, default: Any) -> Any:
    """Read an attribute or dictionary key from faster-whisper objects."""

    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _optional_float(value: Any) -> float | None:
    """Convert optional numeric values to floats."""

    return None if value is None else float(value)

