"""Command-line entrypoint for running the pipeline, review services, and UI."""

import argparse
import asyncio

import uvicorn

from yt_pipeline.config import get_settings
from yt_pipeline.database import build_repository
from yt_pipeline.downloader import YouTubeDownloader
from yt_pipeline.pipeline import VideoPipeline
from yt_pipeline.scheduler import start_scheduler
from yt_pipeline.stages.ai_review import AIReviewService, RankingService
from yt_pipeline.ui import create_app


def main() -> None:
    """Parse CLI arguments and run the requested command."""

    parser = argparse.ArgumentParser(description="Run the YouTube pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("run-once")
    subparsers.add_parser("scheduler")
    subparsers.add_parser("ui")
    review_video = subparsers.add_parser("review-video")
    review_video.add_argument("video_id")
    review_video.add_argument("--force", action="store_true")
    review_video.add_argument("--limit", type=int)
    review_pending = subparsers.add_parser("review-pending")
    review_pending.add_argument("--force", action="store_true")
    review_pending.add_argument("--limit", type=int)
    review_reel = subparsers.add_parser("review-reel")
    review_reel.add_argument("reel_id")
    review_reel.add_argument("--force", action="store_true")
    rank_video = subparsers.add_parser("rank-video")
    rank_video.add_argument("video_id")
    retry = subparsers.add_parser("retry-ai-review")
    retry.add_argument("reel_id")
    assets = subparsers.add_parser("build-review-assets")
    assets.add_argument("reel_id")
    assets.add_argument("--force", action="store_true")
    subparsers.add_parser("ai-health")
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
        uvicorn.run(create_app(repo, settings), host=settings.ui_host, port=settings.ui_port)
    elif args.command == "review-video":
        result = asyncio.run(AIReviewService(repo, settings).review_video(args.video_id, force=args.force, limit=args.limit))
        print_review_summary(result)
    elif args.command == "review-pending":
        result = asyncio.run(AIReviewService(repo, settings).review_pending(force=args.force, limit=args.limit))
        print_review_summary(result)
    elif args.command == "review-reel":
        video, _reel = repo.find_reel(args.reel_id)
        result = asyncio.run(AIReviewService(repo, settings).review_reel(video.video_id, args.reel_id, force=args.force))
        print(result.model_dump_json(by_alias=True, indent=2))
    elif args.command == "rank-video":
        print(RankingService(repo, settings).rank_video(args.video_id))
    elif args.command == "retry-ai-review":
        video, _reel = repo.find_reel(args.reel_id)
        result = asyncio.run(AIReviewService(repo, settings).review_reel(video.video_id, args.reel_id, force=True))
        print(result.model_dump_json(by_alias=True, indent=2))
    elif args.command == "build-review-assets":
        video, reel = repo.find_reel(args.reel_id)
        assets_data = AIReviewService(repo, settings).asset_builder.build(video.video_id, reel, force=args.force)
        repo.update_reel(video.video_id, args.reel_id, {"reviewAssets": assets_data})
        print(f"Built review assets for {args.reel_id}.")
    elif args.command == "ai-health":
        print(asyncio.run(AIReviewService(repo, settings).ollama.health()))


def print_review_summary(result: dict) -> None:
    """Print a concise review-video command summary."""

    if "videos" in result:
        print(f"Videos: {result.get('videos', 0)}")
    print(f"Reviewed: {result.get('reviewed', 0)}")
    print(f"Failed: {result.get('failed', 0)}")
    print(f"Ranked: {result.get('ranked', 0)}")
    if result.get("topReel"):
        print(f"Top reel: {result['topReel']}")


if __name__ == "__main__":
    main()
