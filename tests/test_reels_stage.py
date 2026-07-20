"""Unit tests for vertical reel generation."""

import tempfile
import unittest
from pathlib import Path

from yt_pipeline.models import ReelGenerationResultDTO, StageName, StageState, VideoDocument, VideoStatus
from yt_pipeline.stages.reels import ReelsStage, VERTICAL_FILTER


class FakeFFmpegRunner:
    """Fake FFmpeg runner that records clip creation requests."""

    def __init__(self, duration: float) -> None:
        """Create a fake runner with a fixed source duration."""

        self.duration = duration
        self.calls: list[dict] = []

    def probe_duration(self, video_path: Path) -> float:
        """Return the configured fake duration."""

        return self.duration

    def create_vertical_clip(self, *, source: Path, output: Path, start: float, duration: float) -> None:
        """Record the requested clip and create a fake output file."""

        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"fake mp4")
        self.calls.append({"source": source, "output": output, "start": start, "duration": duration})


class ReelsStageTests(unittest.TestCase):
    """Tests for reel clip generation and metadata."""

    def test_vertical_filter_center_crops_to_reel_size(self) -> None:
        """FFmpeg filter scales and center-crops output to 1080x1920."""

        self.assertEqual(
            VERTICAL_FILTER,
            "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920",
        )

    def test_process_generates_bounded_vertical_clip_candidates(self) -> None:
        """Stage splits videos into bounded reel candidates with score placeholders."""

        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.mp4"
            source.write_bytes(b"fake source")
            video = VideoDocument(videoId="abc", title="Video", localPath=str(source), status=VideoStatus.TRANSCRIBED)
            video.stages[StageName.TRANSCRIPTION.value] = StageState(completed=True)
            ffmpeg = FakeFFmpegRunner(duration=75)

            result = ReelsStage(Path(temp_dir) / "reels", clip_seconds=30, max_clips=2, ffmpeg=ffmpeg).process(video)

            self.assertIsInstance(result, ReelGenerationResultDTO)
            self.assertEqual(result.total_generated, 2)
            self.assertEqual([call["start"] for call in ffmpeg.calls], [0.0, 30.0])
            self.assertEqual(result.clips[0].width, 1080)
            self.assertEqual(result.clips[0].height, 1920)
            self.assertTrue(result.clips[0].path.exists())

    def test_process_rejects_missing_source_file(self) -> None:
        """Stage fails clearly when the downloaded source is missing."""

        video = VideoDocument(videoId="abc", title="Video", localPath="/missing.mp4", status=VideoStatus.TRANSCRIBED)

        with self.assertRaises(FileNotFoundError):
            ReelsStage(Path("/tmp/reels"), ffmpeg=FakeFFmpegRunner(duration=30)).process(video)


if __name__ == "__main__":
    unittest.main()

