"""AI review stage implementation."""

from yt_pipeline.models import VideoDocument


def review_reels(video: VideoDocument) -> dict[str, int]:
    """Return a minimal review summary for generated reels."""

    generated = video.stages.get("reels")
    total = int(generated.metadata.get("totalGenerated", 0)) if generated else 0
    return {"reviewed": total, "approved": total, "rejected": 0}

