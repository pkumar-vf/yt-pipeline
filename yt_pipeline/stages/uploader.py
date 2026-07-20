"""Upload stage implementation."""

from yt_pipeline.models import VideoDocument


def upload_ready_reels(video: VideoDocument) -> list[str]:
    """Return uploaded video identifiers for reels ready to publish."""

    clips = video.stages.get("reels")
    return list(clips.metadata.get("clips", [])) if clips else []

