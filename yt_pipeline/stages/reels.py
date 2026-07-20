"""Vertical reel generation stage implemented with FFmpeg."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from yt_pipeline.models import ReelClipDTO, ReelGenerationResultDTO, VideoDocument

LOGGER = logging.getLogger(__name__)
VERTICAL_FILTER = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920"


class FFmpegRunner:
    """Small wrapper around FFmpeg and FFprobe command execution."""

    def probe_duration(self, video_path: Path) -> float:
        """Return the input video duration in seconds using FFprobe."""

        command = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(video_path),
        ]
        completed = subprocess.run(command, capture_output=True, check=True, text=True)
        payload = json.loads(completed.stdout)
        return float(payload["format"]["duration"])

    def create_vertical_clip(self, *, source: Path, output: Path, start: float, duration: float) -> None:
        """Create one centered 1080x1920 MP4 clip from the source video."""

        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(source),
            "-vf",
            VERTICAL_FILTER,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(output),
        ]
        subprocess.run(command, capture_output=True, check=True, text=True)


class ReelsStage:
    """Generate vertical reel candidates from a downloaded source video."""

    def __init__(
        self,
        output_dir: Path,
        *,
        clip_seconds: int = 30,
        max_clips: int = 20,
        ffmpeg: FFmpegRunner | None = None,
        logger: logging.Logger = LOGGER,
    ) -> None:
        """Create a reel stage with configurable clip sizing and limits."""

        self.output_dir = output_dir
        self.clip_seconds = clip_seconds
        self.max_clips = max_clips
        self.ffmpeg = ffmpeg or FFmpegRunner()
        self.logger = logger

    def process(self, video: VideoDocument) -> ReelGenerationResultDTO:
        """Create centered 9:16 reel candidates for one video."""

        source = self._validate_source(video)
        duration = self.ffmpeg.probe_duration(source)
        clips: list[ReelClipDTO] = []
        for index, start in enumerate(self._clip_starts(duration), start=1):
            end = min(start + self.clip_seconds, duration)
            output = self.output_dir / video.video_id / f"{video.video_id}-{index:03}.mp4"
            self.logger.info("Creating vertical reel %s from %.2fs to %.2fs", output, start, end)
            self.ffmpeg.create_vertical_clip(source=source, output=output, start=start, duration=end - start)
            clips.append(
                ReelClipDTO(
                    id=f"{video.video_id}-{index:03}",
                    path=output,
                    start=start,
                    end=end,
                    duration=end - start,
                )
            )

        self.logger.info("Generated %s vertical reels for video_id=%s", len(clips), video.video_id)
        return ReelGenerationResultDTO(clips=clips, totalGenerated=len(clips), clipSeconds=self.clip_seconds)

    def _validate_source(self, video: VideoDocument) -> Path:
        """Return the local video path after validating it exists."""

        if not video.local_path:
            raise ValueError("Video document is missing localPath.")
        source = Path(video.local_path)
        if not source.exists():
            raise FileNotFoundError(f"Local video path does not exist: {source}")
        return source

    def _clip_starts(self, duration: float) -> list[float]:
        """Return bounded sequential clip start times for the source duration."""

        if duration <= 0:
            return []
        starts: list[float] = []
        start = 0.0
        while start < duration and len(starts) < self.max_clips:
            starts.append(start)
            start += self.clip_seconds
        return starts


def generate_reels(
    video: VideoDocument,
    output_dir: Path,
    *,
    clip_seconds: int = 30,
    max_clips: int = 20,
) -> ReelGenerationResultDTO:
    """Generate vertical reel candidates using the default FFmpeg runner."""

    return ReelsStage(output_dir, clip_seconds=clip_seconds, max_clips=max_clips).process(video)

