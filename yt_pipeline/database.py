"""MongoDB access layer for video pipeline state."""

from collections.abc import Iterable
from typing import Any

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.collection import Collection

from yt_pipeline.models import StageName, StageState, VideoDocument, VideoStatus, utc_now


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


def build_repository(mongo_uri: str, database: str, collection: str) -> VideoRepository:
    """Create a Mongo-backed repository from connection settings."""

    client: MongoClient = MongoClient(mongo_uri)
    repo = VideoRepository(client[database][collection])
    repo.ensure_indexes()
    return repo
