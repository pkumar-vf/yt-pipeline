"""Unit tests for YouTube downloader options."""

import unittest
from pathlib import Path
from unittest.mock import patch

from yt_pipeline.downloader import BEST_AUDIO_FORMAT, BEST_AVAILABLE_FORMAT, YouTubeDownloader
from yt_pipeline.models import DiscoveredVideoDTO


class FakeYoutubeDL:
    """Small yt-dlp fake that captures options and returns deterministic metadata."""

    options: list[dict] = []

    def __init__(self, options: dict) -> None:
        """Capture yt-dlp options for assertions."""

        self.options = options
        FakeYoutubeDL.options.append(options)

    def __enter__(self) -> "FakeYoutubeDL":
        """Return this fake as a context manager value."""

        return self

    def __exit__(self, *args: object) -> None:
        """Leave the fake context manager."""

    def extract_info(self, url: str, download: bool) -> dict:
        """Return metadata shaped like a merged yt-dlp download."""

        return {
            "id": "abc",
            "title": "Video",
            "requested_formats": [{"format_id": "v"}, {"format_id": "a"}],
            "requested_downloads": [{"filepath": self._filepath()}],
        }

    def prepare_filename(self, info: dict) -> str:
        """Return yt-dlp's pre-merge filename fallback."""

        return "downloads/abc.webm"

    def _filepath(self) -> str:
        """Return the fake output path for video or audio options."""

        if self.options["format"] == BEST_AUDIO_FORMAT:
            return "downloads/audio/abc.webm"
        return "downloads/video/abc.mp4"


class YouTubeDownloaderTests(unittest.TestCase):
    """Tests for download format selection and output path handling."""

    def test_download_uses_highest_available_video_and_audio(self) -> None:
        """Downloader asks yt-dlp for best video plus best audio."""

        video = DiscoveredVideoDTO(video_id="abc", title="Video", webpage_url="https://youtu.be/abc")
        FakeYoutubeDL.options = []
        with patch("yt_pipeline.downloader.yt_dlp.YoutubeDL", FakeYoutubeDL):
            downloaded = YouTubeDownloader(Path("downloads")).download(video)

        self.assertEqual(FakeYoutubeDL.options[0]["format"], BEST_AVAILABLE_FORMAT)
        self.assertEqual(FakeYoutubeDL.options[0]["merge_output_format"], "mp4")
        self.assertEqual(FakeYoutubeDL.options[1]["format"], BEST_AUDIO_FORMAT)
        self.assertEqual(FakeYoutubeDL.options[1]["postprocessors"][0]["preferredcodec"], "m4a")
        self.assertEqual(downloaded.local_path, Path("downloads/video/abc.mp4"))
        self.assertEqual(downloaded.audio_path, Path("downloads/audio/abc.m4a"))


if __name__ == "__main__":
    unittest.main()
