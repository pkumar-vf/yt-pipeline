"""Command-line entrypoint for running the pipeline and UI."""

import argparse

import uvicorn

from yt_pipeline.config import get_settings
from yt_pipeline.database import build_repository
from yt_pipeline.downloader import YouTubeDownloader
from yt_pipeline.pipeline import VideoPipeline
from yt_pipeline.scheduler import start_scheduler
from yt_pipeline.ui import create_app


def main() -> None:
    """Parse CLI arguments and run the requested command."""

    parser = argparse.ArgumentParser(description="Run the YouTube pipeline.")
    parser.add_argument("command", choices=("run-once", "scheduler", "ui"))
    args = parser.parse_args()

    settings = get_settings()
    settings.ensure_dirs()
    repo = build_repository(settings.mongo_uri, settings.mongo_db, settings.videos_collection)

    if args.command == "run-once":
        pipeline = VideoPipeline(settings, repo, YouTubeDownloader(settings.downloads_dir))
        created = pipeline.run_once()
        print(f"Downloaded {created} new video(s).")
    elif args.command == "scheduler":
        start_scheduler(settings, repo)
    elif args.command == "ui":
        uvicorn.run(create_app(repo), host=settings.ui_host, port=settings.ui_port)


if __name__ == "__main__":
    main()

