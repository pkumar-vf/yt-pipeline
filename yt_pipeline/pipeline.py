"""Pipeline orchestration for discovery, download, and processing stages."""

from yt_pipeline.config import Settings
from yt_pipeline.database import VideoRepository
from yt_pipeline.downloader import YouTubeDownloader
from yt_pipeline.models import StageName, VideoDocument, VideoStatus
from yt_pipeline.stages.ai_review import review_reels
from yt_pipeline.stages.reels import generate_reels
from yt_pipeline.stages.transcription import TranscriptionStage
from yt_pipeline.stages.uploader import upload_ready_reels


class VideoPipeline:
    """Coordinates each pipeline step while persistence stays in the repository."""

    def __init__(self, settings: Settings, repo: VideoRepository, downloader: YouTubeDownloader) -> None:
        """Create a pipeline from settings and its external dependencies."""

        self.settings = settings
        self.repo = repo
        self.downloader = downloader
        self.transcription_stage = TranscriptionStage(repo, settings.transcripts_dir)

    def discover_and_download(self, *, limit: int = 10) -> int:
        """Discover channel videos, download new ones, and store stage state."""

        if not self.settings.channel_url:
            raise ValueError("YT_CHANNEL_URL is required before discovery can run.")

        created = 0
        for discovered in self.downloader.discover(self.settings.channel_url, limit=limit):
            if self.repo.exists(discovered.video_id):
                continue

            video = VideoDocument(
                videoId=discovered.video_id,
                title=discovered.title,
                publishedAt=discovered.published_at,
            )
            self.repo.insert_discovered(video)
            self.repo.set_status(discovered.video_id, VideoStatus.DOWNLOADING)

            try:
                downloaded = self.downloader.download(discovered)
                self.repo.complete_stage(
                    discovered.video_id,
                    StageName.DOWNLOAD,
                    status=VideoStatus.DOWNLOADED,
                    path=str(downloaded.local_path),
                    local_path=str(downloaded.local_path),
                    audio_path=str(downloaded.audio_path),
                    metadata=downloaded.metadata,
                )
                created += 1
            except Exception as exc:
                self.repo.fail_stage(discovered.video_id, StageName.DOWNLOAD, str(exc))
        return created

    def process_ready(self) -> int:
        """Run local processing stages until no eligible video advances."""

        advanced = 0
        while True:
            progressed = self._process_next_batch()
            advanced += progressed
            if progressed == 0:
                return advanced

    def _process_next_batch(self) -> int:
        """Advance each ready video by one stage and return the progress count."""

        progressed = 0
        statuses = [
            VideoStatus.DOWNLOADED,
            VideoStatus.TRANSCRIBED,
            VideoStatus.REELS_GENERATED,
            VideoStatus.AI_REVIEWED,
        ]
        for video in self.repo.list_by_status(statuses, limit=100):
            if video.status == VideoStatus.DOWNLOADED:
                self._transcribe(video)
                progressed += 1
            elif video.status == VideoStatus.TRANSCRIBED:
                self._generate_reels(video)
                progressed += 1
            elif video.status == VideoStatus.REELS_GENERATED:
                self._review(video)
                progressed += 1
            elif video.status == VideoStatus.AI_REVIEWED:
                self.repo.set_status(video.video_id, VideoStatus.READY_FOR_UPLOAD)
                progressed += 1
        return progressed

    def upload_ready(self) -> None:
        """Upload videos that have passed all pre-upload stages."""

        for video in self.repo.list_recent(limit=100):
            if video.status != VideoStatus.READY_FOR_UPLOAD:
                continue
            uploaded = upload_ready_reels(video)
            self.repo.complete_stage(
                video.video_id,
                StageName.INSTAGRAM,
                status=VideoStatus.UPLOADED,
                metadata={"uploadedVideos": uploaded},
            )

    def run_once(self) -> int:
        """Run one complete scheduler tick and return the number of new downloads."""

        created = self.discover_and_download()
        self.process_ready()
        self.upload_ready()
        return created

    def _transcribe(self, video: VideoDocument) -> None:
        """Run transcription for a downloaded video."""

        try:
            self.transcription_stage.process(video.video_id)
        except Exception as exc:
            self.repo.fail_stage(video.video_id, StageName.TRANSCRIPTION, str(exc))

    def _generate_reels(self, video: VideoDocument) -> None:
        """Run reel generation for a transcribed video."""

        try:
            result = generate_reels(
                video,
                self.settings.reels_dir,
                clip_seconds=self.settings.reel_clip_seconds,
                max_clips=self.settings.reel_max_clips,
            )
            self.repo.complete_stage(
                video.video_id,
                StageName.REELS,
                status=VideoStatus.REELS_GENERATED,
                metadata={
                    "width": 1080,
                    "height": 1920,
                    "aspectRatio": "9:16",
                    "clipSeconds": result.clip_seconds,
                    "totalGenerated": result.total_generated,
                    "clips": [clip.model_dump(mode="json") for clip in result.clips],
                    "ranking": [
                        {
                            "clipId": clip.id,
                            "clipPath": str(clip.path),
                            "score": None,
                            "rank": None,
                            "reviewStatus": "pending",
                        }
                        for clip in result.clips
                    ],
                },
            )
        except Exception as exc:
            self.repo.fail_stage(video.video_id, StageName.REELS, str(exc))

    def _review(self, video: VideoDocument) -> None:
        """Run AI review for generated reels."""

        try:
            review = review_reels(video)
            self.repo.complete_stage(
                video.video_id,
                StageName.AI_REVIEW,
                status=VideoStatus.AI_REVIEWED,
                metadata=review,
            )
        except Exception as exc:
            self.repo.fail_stage(video.video_id, StageName.AI_REVIEW, str(exc))
