"""Unit tests for the faster-whisper transcription stage."""

from __future__ import annotations

import json
import logging
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from yt_pipeline.models import StageName, StageState, VideoDocument, VideoStatus
from yt_pipeline.stages.transcription import (
    SrtExporter,
    TranscriptJsonStreamer,
    TranscriptionStage,
    WhisperModelProvider,
    format_srt_timestamp,
)


@dataclass
class FakeWord:
    """Small faster-whisper word stand-in used by unit tests."""

    word: str
    start: float
    end: float
    probability: float


@dataclass
class FakeSegment:
    """Small faster-whisper segment stand-in used by unit tests."""

    id: int
    start: float
    end: float
    text: str
    words: list[FakeWord]


@dataclass
class FakeInfo:
    """Small faster-whisper metadata stand-in used by unit tests."""

    language: str
    duration: float


class FakeModel:
    """Fake Whisper model that records transcribe arguments."""

    def __init__(self, segments: list[FakeSegment]) -> None:
        """Create a fake model with deterministic segments."""

        self.segments = segments
        self.calls: list[dict[str, Any]] = []

    def transcribe(self, audio: str, **kwargs: Any) -> tuple[list[FakeSegment], FakeInfo]:
        """Return fake segments and capture the faster-whisper options."""

        self.calls.append({"audio": audio, **kwargs})
        return self.segments, FakeInfo(language="en", duration=12.5)


class FakeProvider:
    """Singleton-like fake model provider for stage tests."""

    model: FakeModel
    calls = 0

    @classmethod
    def get(cls, model_name: str = "large-v3") -> FakeModel:
        """Return the configured fake model and track provider calls."""

        cls.calls += 1
        return cls.model


class FakeRepository:
    """In-memory repository implementing the methods used by TranscriptionStage."""

    def __init__(self, video: VideoDocument | None) -> None:
        """Create a fake repository with an optional video."""

        self.video = video
        self.completed: dict[str, Any] | None = None
        self.failed: tuple[str, str] | None = None

    def get(self, video_id: str) -> VideoDocument | None:
        """Return the configured video when ids match."""

        if self.video and self.video.video_id == video_id:
            return self.video
        return None

    def complete_transcription(self, video_id: str, **kwargs: Any) -> None:
        """Capture a successful transcription update."""

        self.completed = {"video_id": video_id, **kwargs}

    def fail_transcription(self, video_id: str, error: str) -> None:
        """Capture a failed transcription update."""

        self.failed = (video_id, error)


class TranscriptionStageTests(unittest.TestCase):
    """Behavior tests for validation, export, and Mongo update flow."""

    def test_format_srt_timestamp(self) -> None:
        """SRT timestamps include hours, minutes, seconds, and milliseconds."""

        self.assertEqual(format_srt_timestamp(3723.456), "01:02:03,456")

    def test_streamer_writes_json_and_srt(self) -> None:
        """Transcript exporter writes the required JSON shape and SRT cues."""

        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "abc.json"
            srt_path = Path(temp_dir) / "abc.srt"
            segment = FakeSegment(
                id=0,
                start=1.25,
                end=2.5,
                text=" Hello ",
                words=[FakeWord("Hello", 1.25, 2.5, 0.98)],
            )
            count = TranscriptJsonStreamer(json_path).write(
                metadata=type("Meta", (), {"language": "en", "duration": 2.5})(),
                segments=[segment],
                srt_path=srt_path,
                srt_exporter=SrtExporter(),
                progress_callback=lambda _: None,
            )

            data = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(count, 1)
            self.assertEqual(data["language"], "en")
            self.assertEqual(data["segments"][0]["words"][0]["probability"], 0.98)
            self.assertIn("00:00:01,250 --> 00:00:02,500", srt_path.read_text(encoding="utf-8"))

    def test_process_updates_mongo_after_transcription(self) -> None:
        """Stage transcribes valid videos and writes Mongo update fields."""

        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "video.mp4"
            video_path.write_bytes(b"fake")
            video = VideoDocument(videoId="abc", title="A", localPath=str(video_path), status=VideoStatus.DOWNLOADED)
            video.stages[StageName.DOWNLOAD.value] = StageState(completed=True)
            repo = FakeRepository(video)
            FakeProvider.model = FakeModel(
                [FakeSegment(0, 0.0, 1.0, "Hello", [FakeWord("Hello", 0.0, 1.0, 0.9)])]
            )
            FakeProvider.calls = 0

            stage = TranscriptionStage(
                repo, Path(temp_dir) / "transcripts", model_provider=FakeProvider, logger=logging.getLogger("test")
            )
            stage.process("abc")

            self.assertIsNone(repo.failed)
            self.assertEqual(repo.completed["language"], "en")
            self.assertEqual(repo.completed["total_segments"], 1)
            self.assertEqual(repo.completed["model"], "large-v3")
            self.assertTrue(Path(repo.completed["transcript_path"]).exists())
            self.assertTrue(Path(repo.completed["subtitle_path"]).exists())
            self.assertEqual(FakeProvider.model.calls[0]["beam_size"], 5)
            self.assertTrue(FakeProvider.model.calls[0]["word_timestamps"])
            self.assertTrue(FakeProvider.model.calls[0]["vad_filter"])

    def test_completed_transcription_returns_immediately(self) -> None:
        """Completed videos do not load the model again."""

        video = VideoDocument(videoId="abc", title="A", localPath="/tmp/video.mp4", status=VideoStatus.TRANSCRIBED)
        video.stages[StageName.DOWNLOAD.value] = StageState(completed=True)
        video.stages[StageName.TRANSCRIPTION.value] = StageState(completed=True)
        repo = FakeRepository(video)
        FakeProvider.calls = 0

        TranscriptionStage(repo, Path("/tmp/transcripts"), model_provider=FakeProvider).process("abc")

        self.assertIsNone(repo.completed)
        self.assertIsNone(repo.failed)
        self.assertEqual(FakeProvider.calls, 0)

    def test_validation_error_is_saved_without_raising(self) -> None:
        """Invalid videos store transcription errors without crashing callers."""

        video = VideoDocument(videoId="abc", title="A", localPath="/missing.mp4", status=VideoStatus.DOWNLOADED)
        video.stages[StageName.DOWNLOAD.value] = StageState(completed=True)
        repo = FakeRepository(video)
        logger = logging.getLogger("test.validation")
        logger.disabled = True

        TranscriptionStage(repo, Path("/tmp/transcripts"), model_provider=FakeProvider, logger=logger).process("abc")

        self.assertIsNotNone(repo.failed)
        self.assertIn("does not exist", repo.failed[1])


class WhisperModelProviderTests(unittest.TestCase):
    """Tests for singleton model provider behavior."""

    def test_get_reuses_loaded_model(self) -> None:
        """Provider returns the loaded model until the model name changes."""

        original_model = WhisperModelProvider._model
        original_name = WhisperModelProvider._model_name
        try:
            sentinel = object()
            WhisperModelProvider._model = sentinel
            WhisperModelProvider._model_name = "large-v3"
            self.assertIs(WhisperModelProvider.get("large-v3"), sentinel)
        finally:
            WhisperModelProvider._model = original_model
            WhisperModelProvider._model_name = original_name


if __name__ == "__main__":
    unittest.main()
