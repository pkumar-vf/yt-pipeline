"""Reel generation stage implementation."""

from pathlib import Path

from yt_pipeline.models import VideoDocument


def generate_reels(video: VideoDocument, output_dir: Path) -> list[Path]:
    """Create minimal reel manifest entries for a downloaded video."""

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{video.video_id}-reels.txt"
    manifest_path.write_text(
        f"Reel generation pending for {video.title}\nSource: {video.local_path or 'unknown'}\n",
        encoding="utf-8",
    )
    return [manifest_path]

