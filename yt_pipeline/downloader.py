"""YouTube discovery and download helpers backed by yt-dlp."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yt_dlp

from yt_pipeline.models import DiscoveredVideoDTO, DownloadedVideoDTO

BEST_AVAILABLE_FORMAT = "bv*+ba/b"
BEST_AUDIO_FORMAT = "ba/bestaudio/b"


class YouTubeDownloader:
    """Thin wrapper around yt-dlp for channel discovery and video downloads."""

    def __init__(self, downloads_dir: Path) -> None:
        """Create a downloader that writes files into the given directory."""

        self.downloads_dir = downloads_dir

    def discover(self, channel_url: str, *, limit: int = 10) -> list[DiscoveredVideoDTO]:
        """Return recent videos from a YouTube channel without downloading them."""

        options = {"extract_flat": True, "quiet": True, "playlistend": limit}
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(channel_url, download=False)

        entries = info.get("entries", []) if info else []
        return [self._to_discovered(entry) for entry in entries if entry.get("id")]

    def download(self, video: DiscoveredVideoDTO) -> DownloadedVideoDTO:
        """Download highest-quality video plus a separate audio file."""

        self.downloads_dir.mkdir(parents=True, exist_ok=True)
        video_dir = self.downloads_dir / "video"
        audio_dir = self.downloads_dir / "audio"
        video_dir.mkdir(parents=True, exist_ok=True)
        audio_dir.mkdir(parents=True, exist_ok=True)

        video_options = {
            "outtmpl": str(video_dir / "%(id)s.%(ext)s"),
            "quiet": True,
            "format": BEST_AVAILABLE_FORMAT,
            "merge_output_format": "mp4",
        }
        audio_options = {
            "outtmpl": str(audio_dir / "%(id)s.%(ext)s"),
            "quiet": True,
            "format": BEST_AUDIO_FORMAT,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "m4a",
                }
            ],
        }

        with yt_dlp.YoutubeDL(video_options) as ydl:
            info = ydl.extract_info(video.webpage_url, download=True)
            path = self._downloaded_path(ydl, info)
        with yt_dlp.YoutubeDL(audio_options) as ydl:
            audio_info = ydl.extract_info(video.webpage_url, download=True)
            audio_path = self._audio_path(ydl, audio_info)

        return DownloadedVideoDTO(
            video_id=video.video_id,
            title=info.get("title") or video.title,
            local_path=path,
            audio_path=audio_path,
            metadata={**self._metadata(info), "audioPath": str(audio_path)},
        )

    def _downloaded_path(self, ydl: yt_dlp.YoutubeDL, info: dict[str, Any]) -> Path:
        """Return the final output path after yt-dlp downloads and merges streams."""

        requested_downloads = info.get("requested_downloads") or []
        if requested_downloads and requested_downloads[0].get("filepath"):
            return Path(requested_downloads[0]["filepath"])

        path = Path(ydl.prepare_filename(info))
        if info.get("requested_formats") and path.suffix != ".mp4":
            return path.with_suffix(".mp4")
        return path

    def _audio_path(self, ydl: yt_dlp.YoutubeDL, info: dict[str, Any]) -> Path:
        """Return the final audio path after yt-dlp extracts audio."""

        requested_downloads = info.get("requested_downloads") or []
        if requested_downloads and requested_downloads[0].get("filepath"):
            filepath = Path(requested_downloads[0]["filepath"])
            return filepath if filepath.suffix == ".m4a" else filepath.with_suffix(".m4a")
        return Path(ydl.prepare_filename(info)).with_suffix(".m4a")

    def _to_discovered(self, entry: dict[str, Any]) -> DiscoveredVideoDTO:
        """Map one yt-dlp channel entry into a validated DTO."""

        video_id = entry["id"]
        return DiscoveredVideoDTO(
            video_id=video_id,
            title=entry.get("title") or video_id,
            webpage_url=entry.get("url") or f"https://www.youtube.com/watch?v={video_id}",
            published_at=self._parse_upload_date(entry.get("upload_date")),
            metadata=self._metadata(entry),
        )

    def _metadata(self, data: dict[str, Any]) -> dict[str, Any]:
        """Keep only stable metadata useful for stage inspection."""

        keys = ("title", "channel", "thumbnail", "description", "upload_date", "duration")
        return {key: data.get(key) for key in keys if data.get(key) is not None}

    def _parse_upload_date(self, value: str | None) -> datetime | None:
        """Parse yt-dlp upload dates such as 20260720 into UTC datetimes."""

        if not value:
            return None
        return datetime.strptime(value, "%Y%m%d").replace(tzinfo=timezone.utc)
