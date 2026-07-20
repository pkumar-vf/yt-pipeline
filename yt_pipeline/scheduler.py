"""APScheduler entrypoint for recurring pipeline runs."""

from apscheduler.schedulers.blocking import BlockingScheduler

from yt_pipeline.config import Settings
from yt_pipeline.database import VideoRepository
from yt_pipeline.downloader import YouTubeDownloader
from yt_pipeline.pipeline import VideoPipeline


def start_scheduler(settings: Settings, repo: VideoRepository) -> None:
    """Start a blocking scheduler that runs the pipeline at a fixed interval."""

    pipeline = VideoPipeline(settings, repo, YouTubeDownloader(settings.downloads_dir))
    scheduler = BlockingScheduler()
    scheduler.add_job(
        pipeline.run_once,
        trigger="interval",
        minutes=settings.scheduler_minutes,
        id="youtube-pipeline",
        replace_existing=True,
    )
    scheduler.start()

