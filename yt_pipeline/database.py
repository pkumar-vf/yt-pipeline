"""MongoDB access layer for video pipeline state."""

from collections.abc import Iterable
from typing import Any

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.collection import Collection

from yt_pipeline.models import HumanReviewUpdateDTO, StageName, StageState, VideoDocument, VideoStatus, utc_now


class VideoRepository:
    """Small repository that owns all MongoDB reads and writes for videos."""

    def __init__(self, collection: Collection) -> None:
        """Create a repository for a MongoDB videos collection."""

        self.collection = collection

    def ensure_indexes(self) -> None:
        """Create indexes required for idempotent discovery and fast UI reads."""

        self.collection.create_index([("videoId", ASCENDING)], unique=True)
        self.collection.create_index([("updatedAt", DESCENDING)])
        self.collection.create_index([("status", ASCENDING)])

    def exists(self, video_id: str) -> bool:
        """Return whether a video has already been discovered."""

        return self.collection.count_documents({"videoId": video_id}, limit=1) > 0

    def insert_discovered(self, video: VideoDocument) -> None:
        """Insert a newly discovered video if it is not already present."""

        self.collection.update_one(
            {"videoId": video.video_id},
            {"$setOnInsert": video.to_mongo()},
            upsert=True,
        )

    def get(self, video_id: str) -> VideoDocument | None:
        """Fetch one video by YouTube id."""

        data = self.collection.find_one({"videoId": video_id})
        return VideoDocument.model_validate(data) if data else None

    def list_recent(self, limit: int = 50) -> list[VideoDocument]:
        """Return recently updated videos for the dashboard."""

        docs: Iterable[dict[str, Any]] = self.collection.find().sort("updatedAt", DESCENDING).limit(limit)
        return [VideoDocument.model_validate(doc) for doc in docs]

    def list_by_status(self, statuses: list[VideoStatus], *, limit: int = 100) -> list[VideoDocument]:
        """Return recently updated videos whose status is in the given list."""

        values = [status.value for status in statuses]
        docs: Iterable[dict[str, Any]] = (
            self.collection.find({"status": {"$in": values}}).sort("updatedAt", DESCENDING).limit(limit)
        )
        return [VideoDocument.model_validate(doc) for doc in docs]

    def list_videos_with_pending_ai_review(self, *, limit: int = 100) -> list[VideoDocument]:
        """Return videos that have generated reels with missing or failed AI review."""

        docs: Iterable[dict[str, Any]] = (
            self.collection.find({"stages.reels.completed": True}).sort("updatedAt", DESCENDING).limit(limit)
        )
        videos = [VideoDocument.model_validate(doc) for doc in docs]
        return [video for video in videos if has_pending_reels(video)]

    def set_status(self, video_id: str, status: VideoStatus) -> None:
        """Update only the high-level status for a video."""

        self.collection.update_one(
            {"videoId": video_id},
            {"$set": {"status": status.value, "updatedAt": utc_now()}},
        )

    def complete_stage(
        self,
        video_id: str,
        stage: StageName,
        *,
        status: VideoStatus | None = None,
        path: str | None = None,
        metadata: dict[str, Any] | None = None,
        local_path: str | None = None,
        audio_path: str | None = None,
    ) -> None:
        """Mark a stage complete and optionally advance the video status."""

        now = utc_now()
        update: dict[str, Any] = {
            f"stages.{stage.value}": StageState(
                completed=True,
                timestamp=now,
                path=path,
                metadata=metadata or {},
            ).model_dump(mode="python"),
            "updatedAt": now,
        }
        if status:
            update["status"] = status.value
        if local_path:
            update["localPath"] = local_path
        if audio_path:
            update["audioPath"] = audio_path
        self.collection.update_one({"videoId": video_id}, {"$set": update})

    def complete_transcription(
        self,
        video_id: str,
        *,
        transcript_path: str,
        subtitle_path: str,
        language: str,
        total_segments: int,
        model: str,
    ) -> None:
        """Mark transcription complete using the stage schema consumed by the UI."""

        now = utc_now()
        self.collection.update_one(
            {"videoId": video_id},
            {
                "$set": {
                    "status": VideoStatus.TRANSCRIBED.value,
                    "stages.transcription": {
                        "completed": True,
                        "timestamp": now,
                        "transcriptPath": transcript_path,
                        "subtitlePath": subtitle_path,
                        "language": language,
                        "totalSegments": total_segments,
                        "model": model,
                    },
                    "updatedAt": now,
                }
            },
        )

    def fail_stage(self, video_id: str, stage: StageName, error: str) -> None:
        """Record a stage failure and mark the video as failed."""

        self.collection.update_one(
            {"videoId": video_id},
            {
                "$set": {
                    "status": VideoStatus.FAILED.value,
                    f"stages.{stage.value}.completed": False,
                    f"stages.{stage.value}.error": error,
                    "updatedAt": utc_now(),
                }
            },
        )

    def fail_transcription(self, video_id: str, error: str) -> None:
        """Record a transcription failure without raising to callers."""

        self.collection.update_one(
            {"videoId": video_id},
            {
                "$set": {
                    "status": VideoStatus.FAILED.value,
                    "stages.transcription.completed": False,
                    "stages.transcription.error": error,
                    "updatedAt": utc_now(),
                }
            },
        )

    def get_reel(self, video_id: str, reel_id: str) -> tuple[VideoDocument, dict[str, Any]]:
        """Return a video and one reel metadata record by ids."""

        video = self.get(video_id)
        if not video:
            raise ValueError(f"Video not found: {video_id}")
        stage = video.stages.get(StageName.REELS.value)
        reels = stage.metadata.get("clips", []) if stage else []
        for reel in reels:
            if reel.get("id") == reel_id:
                return video, reel
        raise ValueError(f"Reel not found: {reel_id}")

    def find_reel(self, reel_id: str) -> tuple[VideoDocument, dict[str, Any]]:
        """Find a reel by id across videos."""

        data = self.collection.find_one({"stages.reels.metadata.clips.id": reel_id})
        if not data:
            raise ValueError(f"Reel not found: {reel_id}")
        video = VideoDocument.model_validate(data)
        stage = video.stages.get(StageName.REELS.value)
        for reel in stage.metadata.get("clips", []) if stage else []:
            if reel.get("id") == reel_id:
                return video, reel
        raise ValueError(f"Reel not found: {reel_id}")

    def list_reels(self, video_id: str) -> list[dict[str, Any]]:
        """Return all reel records for a video."""

        video = self.get(video_id)
        if not video:
            raise ValueError(f"Video not found: {video_id}")
        stage = video.stages.get(StageName.REELS.value)
        return list(stage.metadata.get("clips", []) if stage else [])

    def update_reel(self, video_id: str, reel_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        """Merge a patch into one reel record and persist the clips list."""

        video = self.get(video_id)
        if not video:
            raise ValueError(f"Video not found: {video_id}")
        stage = video.stages.get(StageName.REELS.value)
        reels = list(stage.metadata.get("clips", []) if stage else [])
        updated: dict[str, Any] | None = None
        for index, reel in enumerate(reels):
            if reel.get("id") == reel_id:
                updated = deep_merge(dict(reel), patch)
                reels[index] = updated
                break
        if updated is None:
            raise ValueError(f"Reel not found: {reel_id}")
        self.update_reels_metadata(video_id, {"clips": reels})
        return updated

    def update_reels_metadata(self, video_id: str, metadata_patch: dict[str, Any]) -> None:
        """Merge a patch into the reels stage metadata."""

        video = self.get(video_id)
        if not video:
            raise ValueError(f"Video not found: {video_id}")
        stage = video.stages.get(StageName.REELS.value)
        metadata = dict(stage.metadata if stage else {})
        metadata.update(metadata_patch)
        self.collection.update_one(
            {"videoId": video_id},
            {"$set": {"stages.reels.metadata": metadata, "updatedAt": utc_now()}},
        )

    def update_ai_review_stage(self, video_id: str, summary: dict[str, Any]) -> None:
        """Persist the video-level AI review stage summary."""

        self.collection.update_one(
            {"videoId": video_id},
            {"$set": {"stages.aiReview": summary, "updatedAt": utc_now()}},
        )

    def update_human_review(self, reel_id: str, update: HumanReviewUpdateDTO) -> dict[str, Any]:
        """Persist a human review decision for one reel."""

        video, _reel = self.find_reel(reel_id)
        patch = {
            "humanReview": {
                **update.model_dump(by_alias=True, mode="json", exclude_none=True),
                "reviewedAt": utc_now().isoformat(),
            }
        }
        return self.update_reel(video.video_id, reel_id, patch)


def deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries while replacing non-dictionary values."""

    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = deep_merge(dict(base[key]), value)
        else:
            base[key] = value
    return base


def has_pending_reels(video: VideoDocument) -> bool:
    """Return whether a video has at least one reel that still needs AI review."""

    stage = video.stages.get(StageName.REELS.value)
    reels = stage.metadata.get("clips", []) if stage else []
    return any((reel.get("aiReview") or {}).get("status") != "COMPLETED" for reel in reels)


def build_repository(mongo_uri: str, database: str, collection: str) -> VideoRepository:
    """Create a Mongo-backed repository from connection settings."""

    client: MongoClient = MongoClient(mongo_uri)
    repo = VideoRepository(client[database][collection])
    repo.ensure_indexes()
    return repo
