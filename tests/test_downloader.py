"""Unit tests for YouTube downloader options."""

import unittest
from pathlib import Path
from unittest.mock import patch

from yt_pipeline.downloader import BEST_AVAILABLE_FORMAT, YouTubeDownloader
from yt_pipeline.models import DiscoveredVideoDTO


class FakeYoutubeDL:
    """Small yt-dlp fake that captures options and returns deterministic metadata."""

    options: dict | None = None

    def __init__(self, options: dict) -> None:
        """Capture yt-dlp options for assertions."""

        self.options = options
        FakeYoutubeDL.options = options

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
            "requested_downloads": [{"filepath": "downloads/abc.mp4"}],
        }

    def prepare_filename(self, info: dict) -> str:
        """Return yt-dlp's pre-merge filename fallback."""

        return "downloads/abc.webm"


class YouTubeDownloaderTests(unittest.TestCase):
    """Tests for download format selection and output path handling."""

    def test_download_uses_highest_available_video_and_audio(self) -> None:
        """Downloader asks yt-dlp for best video plus best audio."""

        video = DiscoveredVideoDTO(video_id="abc", title="Video", webpage_url="https://youtu.be/abc")
        with patch("yt_pipeline.downloader.yt_dlp.YoutubeDL", FakeYoutubeDL):
            downloaded = YouTubeDownloader(Path("downloads")).download(video)

        self.assertEqual(FakeYoutubeDL.options["format"], BEST_AVAILABLE_FORMAT)
        self.assertEqual(FakeYoutubeDL.options["merge_output_format"], "mp4")
        self.assertEqual(downloaded.local_path, Path("downloads/abc.mp4"))


if __name__ == "__main__":
    unittest.main()

