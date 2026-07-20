"""Local AI review and deterministic ranking for generated reels."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from pydantic import ValidationError

from yt_pipeline.config import Settings
from yt_pipeline.database import VideoRepository
from yt_pipeline.models import AIReviewResult, AIReviewStatus, HumanReviewStatus, VideoDocument

LOGGER = logging.getLogger(__name__)
SYSTEM_PROMPT = """You are reviewing a short gaming clip for possible publication as an Instagram Reel.

Evaluate the actual gameplay visible in the supplied chronological frames together with the transcript and activity metrics.
A transcript alone is not sufficient evidence that a clip is good.
A clip with little or no speech may still be excellent when the visual gameplay contains a kill, clutch, squad wipe, escape, surprise, funny failure, strong reaction, or satisfying payoff.
Do not reward generic speech, menus, loading screens, waiting, looting without payoff, repeated footage, or clips that require extensive stream context.

Judge whether:
- the first two seconds provide a hook
- the action is visually understandable
- there is a clear event
- the event has a payoff
- the clip has little dead time
- the clip works without watching the full stream
- the clip ends naturally
- visual and audio quality are acceptable
- a gaming audience would likely continue watching

Return only data matching the supplied JSON schema."""


def utc_iso() -> str:
    """Return the current UTC time in ISO-8601 format."""

    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ProxySettings:
    """Configurable review-proxy encoding settings."""

    max_width: int
    fps: int
    crf: int
    audio_bitrate: str


class FFmpegReviewTools:
    """FFmpeg and FFprobe helpers for review assets and activity metrics."""

    def probe(self, path: Path) -> dict[str, Any]:
        """Return FFprobe metadata for a media file."""

        command = [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
        completed = subprocess.run(command, capture_output=True, check=True, text=True)
        return json.loads(completed.stdout)

    def create_proxy(self, source: Path, output: Path, settings: ProxySettings) -> None:
        """Create a low-resolution H.264 review proxy without touching the source reel."""

        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-vf",
            f"scale='min({settings.max_width},iw)':-2,fps={settings.fps}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            str(settings.crf),
            "-c:a",
            "aac",
            "-b:a",
            settings.audio_bitrate,
            "-movflags",
            "+faststart",
            str(output),
        ]
        subprocess.run(command, capture_output=True, check=True, text=True)

    def extract_frame(self, source: Path, output: Path, timestamp: float) -> None:
        """Extract one JPEG frame at the requested timestamp."""

        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(source),
            "-frames:v",
            "1",
            "-q:v",
            "3",
            str(output),
        ]
        subprocess.run(command, capture_output=True, check=True, text=True)


class ReviewAssetBuilder:
    """Build and validate lightweight review proxies plus representative frames."""

    def __init__(self, root_dir: Path, tools: FFmpegReviewTools, settings: Settings) -> None:
        """Create an asset builder rooted at the configured review-assets directory."""

        self.root_dir = root_dir
        self.tools = tools
        self.settings = settings

    def build(self, video_id: str, reel: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
        """Create or reuse review assets for one reel and return asset metadata."""

        reel_id = str(reel["id"])
        source = Path(str(reel["path"]))
        if not source.exists():
            raise FileNotFoundError(f"Reel file does not exist: {source}")

        asset_dir = self.root_dir / video_id / reel_id
        proxy_path = asset_dir / "review.mp4"
        original_duration = media_duration(self.tools.probe(source)) or float(reel.get("duration", 0.0) or 0.0)
        if force or not self._valid_proxy(proxy_path, original_duration):
            self.tools.create_proxy(source, proxy_path, self._proxy_settings())

        proxy_probe = self.tools.probe(proxy_path)
        proxy_duration = media_duration(proxy_probe)
        frame_timestamps = representative_timestamps(proxy_duration, self.settings.ai_review_frame_count)
        frame_paths = self._extract_frames(proxy_path, frame_timestamps, force=force)
        video_stream = first_video_stream(proxy_probe)
        return {
            "proxyPath": str(proxy_path),
            "proxySizeBytes": proxy_path.stat().st_size,
            "proxyDurationSeconds": proxy_duration,
            "proxyWidth": int(video_stream.get("width", 0)),
            "proxyHeight": int(video_stream.get("height", 0)),
            "proxyFps": fps_from_stream(video_stream),
            "proxyCodec": video_stream.get("codec_name"),
            "proxySha256": sha256_file(proxy_path),
            "framePaths": [str(path) for path in frame_paths],
            "frameTimestamps": frame_timestamps,
            "completedAt": utc_iso(),
        }

    def _valid_proxy(self, proxy_path: Path, original_duration: float) -> bool:
        """Return whether an existing proxy is complete enough to reuse."""

        if not proxy_path.exists() or proxy_path.stat().st_size <= 0:
            return False
        try:
            proxy_duration = media_duration(self.tools.probe(proxy_path))
        except Exception:
            return False
        return abs(proxy_duration - original_duration) <= max(1.0, original_duration * 0.05)

    def _extract_frames(self, proxy_path: Path, timestamps: list[float], *, force: bool) -> list[Path]:
        """Extract representative frames, reusing existing non-empty files."""

        frames_dir = proxy_path.parent / "frames"
        frame_paths = []
        for index, timestamp in enumerate(timestamps, start=1):
            frame_path = frames_dir / f"frame-{index:03}.jpg"
            if force or not frame_path.exists() or frame_path.stat().st_size <= 0:
                self.tools.extract_frame(proxy_path, frame_path, timestamp)
            frame_paths.append(frame_path)
        return frame_paths

    def _proxy_settings(self) -> ProxySettings:
        """Return proxy settings from runtime configuration."""

        return ProxySettings(
            max_width=self.settings.review_proxy_max_width,
            fps=self.settings.review_proxy_fps,
            crf=self.settings.review_proxy_crf,
            audio_bitrate=self.settings.review_proxy_audio_bitrate,
        )


class TranscriptWindowExtractor:
    """Extract transcript text that overlaps a reel's source-video timestamps."""

    def __init__(self, transcripts_dir: Path, context_seconds: int) -> None:
        """Create an extractor for transcript JSON files."""

        self.transcripts_dir = transcripts_dir
        self.context_seconds = context_seconds

    def extract(self, video_id: str, start: float, end: float) -> dict[str, Any]:
        """Return clip and context transcript text for a source time window."""

        path = self.transcripts_dir / f"{video_id}.json"
        if not path.exists():
            return {"clipText": "", "contextBefore": "", "contextAfter": "", "language": "unknown", "segments": []}

        data = json.loads(path.read_text(encoding="utf-8"))
        segments = data.get("segments") or []
        before_start = start - self.context_seconds
        after_end = end + self.context_seconds
        clip_segments = [simple_segment(item) for item in segments if overlaps(float(item.get("start", 0.0)), float(item.get("end", 0.0)), start, end)]
        before = [simple_segment(item) for item in segments if overlaps(float(item.get("start", 0.0)), float(item.get("end", 0.0)), before_start, start)]
        after = [simple_segment(item) for item in segments if overlaps(float(item.get("start", 0.0)), float(item.get("end", 0.0)), end, after_end)]
        return {
            "clipText": " ".join(segment["text"] for segment in clip_segments).strip(),
            "contextBefore": " ".join(segment["text"] for segment in before).strip(),
            "contextAfter": " ".join(segment["text"] for segment in after).strip(),
            "language": data.get("language") or "unknown",
            "segments": clip_segments,
        }


class ActivityMetricCalculator:
    """Calculate lightweight activity metrics from transcript and proxy metadata."""

    def __init__(self, tools: FFmpegReviewTools) -> None:
        """Create a metric calculator backed by FFprobe metadata."""

        self.tools = tools

    def calculate(self, proxy_path: Path, transcript_review: dict[str, Any], reel: dict[str, Any]) -> dict[str, Any]:
        """Return normalized audio and visual activity metrics."""

        probe = self.tools.probe(proxy_path)
        size = proxy_path.stat().st_size if proxy_path.exists() else 0
        duration = media_duration(probe) or float(reel.get("duration", 0.0) or 0.0)
        bitrate = float((probe.get("format") or {}).get("bit_rate") or 0.0)
        speech_coverage = speech_coverage_ratio(transcript_review.get("segments", []), duration)
        activity_hint = clamp01((bitrate / 1_000_000.0) + speech_coverage * 0.25)
        black_frame_ratio = 0.0 if size > 0 else 1.0
        frozen_frame_ratio = 0.0 if size > 0 else 1.0
        return {
            "audio": {
                "meanLoudnessDb": None,
                "peakLoudnessDb": None,
                "loudnessVariance": activity_hint,
                "suddenIncreaseScore": activity_hint,
                "silenceRatio": clamp01(1.0 - speech_coverage),
                "speechCoverage": speech_coverage,
                "audioActivityScore": clamp01((activity_hint + speech_coverage) / 2),
            },
            "visual": {
                "motionScore": activity_hint,
                "sceneChangeCount": 0,
                "sceneChangeIntensity": activity_hint,
                "blackFrameRatio": black_frame_ratio,
                "frozenFrameRatio": frozen_frame_ratio,
                "visualActivityScore": clamp01(activity_hint * (1 - black_frame_ratio) * (1 - frozen_frame_ratio)),
            },
        }


class OllamaClient:
    """Async client for schema-constrained local Ollama review calls."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: int,
        max_retries: int,
        num_ctx: int = 8192,
        logger: logging.Logger = LOGGER,
    ) -> None:
        """Create an Ollama client from runtime settings."""

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.num_ctx = num_ctx
        self.logger = logger

    async def health(self) -> dict[str, Any]:
        """Return reachability and model availability information."""

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            models = [item.get("name") for item in response.json().get("models", [])]
        return {"ok": self.model in models, "model": self.model, "availableModels": models}

    async def review(self, *, prompt: str, image_paths: list[Path]) -> AIReviewResult:
        """Submit a review prompt and chronological images, returning a validated result."""

        images = [base64.b64encode(path.read_bytes()).decode("ascii") for path in image_paths]
        validation_error = ""
        last_error: Exception | None = None
        attempts = max(1, self.max_retries + 1)
        for attempt in range(1, attempts + 1):
            started = time.monotonic()
            try:
                content = prompt if not validation_error else f"{prompt}\n\nFix this schema error: {validation_error}"
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": content, "images": images},
                    ],
                    "format": AIReviewResult.model_json_schema(by_alias=True),
                    "options": {"temperature": 0.1, "num_ctx": self.num_ctx},
                    "stream": False,
                }
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    response = await client.post(f"{self.base_url}/api/chat", json=payload)
                    if response.status_code in {400, 404}:
                        response = await client.post(
                            f"{self.base_url}/api/generate",
                            json={
                                "model": self.model,
                                "system": SYSTEM_PROMPT,
                                "prompt": content,
                                "images": images,
                                "format": AIReviewResult.model_json_schema(by_alias=True),
                                "options": {"temperature": 0.1, "num_ctx": self.num_ctx},
                                "stream": False,
                            },
                        )
                    raise_for_ollama_status(response)
                self.logger.info("Ollama inference completed in %.2fs", time.monotonic() - started)
                return parse_ollama_review_response(response.json())
            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError, ValidationError, json.JSONDecodeError) as exc:
                last_error = exc
                validation_error = str(exc)
                if attempt >= attempts:
                    break
                await asyncio.sleep(0.25 * attempt)
        raise RuntimeError(f"Ollama review failed: {last_error}")


class AIReviewService:
    """Review generated reels through local assets, metrics, transcript, and Ollama."""

    def __init__(self, repo: VideoRepository, settings: Settings, *, ollama: OllamaClient | None = None, tools: FFmpegReviewTools | None = None, logger: logging.Logger = LOGGER) -> None:
        """Create a review service from repository and runtime dependencies."""

        self.repo = repo
        self.settings = settings
        self.tools = tools or FFmpegReviewTools()
        self.asset_builder = ReviewAssetBuilder(settings.review_assets_dir, self.tools, settings)
        self.transcripts = TranscriptWindowExtractor(settings.transcripts_dir, settings.ai_review_transcript_context_seconds)
        self.metrics = ActivityMetricCalculator(self.tools)
        self.ollama = ollama or OllamaClient(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            timeout_seconds=settings.ollama_timeout_seconds,
            max_retries=settings.ollama_max_retries,
            num_ctx=settings.ollama_num_ctx,
            logger=logger,
        )
        self.semaphore = asyncio.Semaphore(max(1, settings.ai_review_concurrency))
        self.logger = logger

    async def review_reel(self, video_id: str, reel_id: str, force: bool = False) -> AIReviewResult:
        """Review one reel idempotently and persist the validated AI result."""

        async with self.semaphore:
            _video, reel = self.repo.get_reel(video_id, reel_id)
            existing = reel.get("aiReview") or {}
            if not force and existing.get("status") == AIReviewStatus.COMPLETED.value:
                return AIReviewResult.model_validate(existing)
            started_at = utc_iso()
            self.repo.update_reel(video_id, reel_id, {"aiReview": {"status": "PROCESSING", "model": self.settings.ollama_model, "startedAt": started_at}})
            try:
                review_assets = self.asset_builder.build(video_id, reel, force=force)
                transcript_review = self.transcripts.extract(video_id, float(reel.get("start", 0.0)), float(reel.get("end", 0.0)))
                metrics = self.metrics.calculate(Path(review_assets["proxyPath"]), transcript_review, reel)
                result = await self.ollama.review(prompt=build_review_prompt(reel, transcript_review, metrics), image_paths=[Path(path) for path in review_assets["framePaths"]])
                self.repo.update_reel(
                    video_id,
                    reel_id,
                    {
                        "reviewAssets": review_assets,
                        "transcriptReview": transcript_review,
                        "activityMetrics": metrics,
                        "humanReview": reel.get("humanReview") or {"status": HumanReviewStatus.PENDING.value},
                        "aiReview": {
                            "status": AIReviewStatus.COMPLETED.value,
                            "model": self.settings.ollama_model,
                            "startedAt": started_at,
                            "completedAt": utc_iso(),
                            "attempts": 1,
                            **result.model_dump(by_alias=True),
                            "error": None,
                        },
                    },
                )
                return result
            except Exception as exc:
                self.logger.exception("AI review failed for reel_id=%s", reel_id)
                self.repo.update_reel(
                    video_id,
                    reel_id,
                    {
                        "aiReview": {
                            "status": AIReviewStatus.FAILED.value,
                            "model": self.settings.ollama_model,
                            "startedAt": started_at,
                            "attempts": int(existing.get("attempts", 0)) + 1,
                            "error": {"type": type(exc).__name__, "message": str(exc), "timestamp": utc_iso(), "retryable": True},
                        }
                    },
                )
                raise

    async def review_video(self, video_id: str, *, force: bool = False, limit: int | None = None) -> dict[str, Any]:
        """Review pending reels for one video and rank successful results."""

        if not self.settings.ai_review_enabled:
            LOGGER.warning("AI review is disabled; leaving reels pending for video_id=%s", video_id)
            return {"reviewed": 0, "failed": 0, "disabled": True}
        video = self.repo.get(video_id)
        if not video:
            raise ValueError(f"Video not found: {video_id}")
        reels = video.stages.get("reels").metadata.get("clips", []) if video.stages.get("reels") else []
        candidates = [reel for reel in reels if force or (reel.get("aiReview") or {}).get("status") != "COMPLETED"]
        if limit:
            candidates = candidates[:limit]
        reviewed = 0
        failed = 0
        for reel in candidates:
            try:
                await self.review_reel(video_id, str(reel["id"]), force=force)
                reviewed += 1
            except Exception:
                failed += 1
        ranking = RankingService(self.repo, self.settings).rank_video(video_id)
        return {"reviewed": reviewed, "failed": failed, **ranking}

    async def review_pending(self, *, force: bool = False, limit: int | None = None) -> dict[str, Any]:
        """Review pending reels across all videos with generated reel candidates."""

        total_reviewed = 0
        total_failed = 0
        total_ranked = 0
        processed_videos = 0
        top_reels: list[str] = []
        remaining = limit
        for video in self.repo.list_videos_with_pending_ai_review():
            if remaining is not None and remaining <= 0:
                break
            result = await self.review_video(video.video_id, force=force, limit=remaining)
            processed_videos += 1
            total_reviewed += int(result.get("reviewed", 0))
            total_failed += int(result.get("failed", 0))
            total_ranked += int(result.get("ranked", 0))
            if result.get("topReel"):
                top_reels.append(str(result["topReel"]))
            if remaining is not None:
                remaining -= int(result.get("reviewed", 0)) + int(result.get("failed", 0))
        return {
            "videos": processed_videos,
            "reviewed": total_reviewed,
            "failed": total_failed,
            "ranked": total_ranked,
            "topReel": top_reels[0] if top_reels else None,
        }


class RankingService:
    """Deterministically rank successfully reviewed reels for a source video."""

    def __init__(self, repo: VideoRepository, settings: Settings) -> None:
        """Create a ranker with duplicate-detection settings."""

        self.repo = repo
        self.settings = settings

    def rank_video(self, video_id: str) -> dict[str, Any]:
        """Rank reviewed reels, persist ranks, and update the video AI stage."""

        video = self.repo.get(video_id)
        if not video:
            raise ValueError(f"Video not found: {video_id}")
        stage = video.stages.get("reels")
        reels = list(stage.metadata.get("clips", [])) if stage else []
        reviewed = [reel for reel in reels if (reel.get("aiReview") or {}).get("status") == "COMPLETED"]
        scored = [self._score(reel, reviewed) for reel in reviewed]
        scored.sort(key=ranking_sort_key)
        total = len(scored)
        for index, reel in enumerate(scored, start=1):
            reel["ranking"]["rank"] = index
            reel["ranking"]["percentile"] = round(((total - index + 1) / total) * 100) if total else 0
            reel["ranking"]["rankedAt"] = utc_iso()
        lookup = {reel["id"]: reel for reel in scored}
        merged = [lookup.get(reel.get("id"), reel) for reel in reels]
        self.repo.update_reels_metadata(video_id, {"clips": merged, "ranking": [dict(reel.get("ranking", {}), reelId=reel["id"]) for reel in scored]})
        self.repo.update_ai_review_stage(video_id, ai_review_summary(merged, self.settings.ollama_model))
        return {"ranked": total, "topReel": scored[0]["id"] if scored else None}

    def _score(self, reel: dict[str, Any], all_reels: list[dict[str, Any]]) -> dict[str, Any]:
        """Return a reel copy with ranking and duplicate-analysis metadata."""

        item = dict(reel)
        scores = (item.get("aiReview") or {}).get("scores") or {}
        raw_score = weighted_score(scores)
        duplicate = duplicate_analysis(item, all_reels, self.settings.ai_review_max_overlap_ratio)
        penalties = activity_penalty(item) + confidence_penalty(item)
        if duplicate["isPossibleDuplicate"]:
            penalties += self.settings.ai_review_duplicate_penalty
            duplicate["penalty"] = self.settings.ai_review_duplicate_penalty
        adjusted = max(0.0, min(10.0, raw_score - penalties))
        item["duplicateAnalysis"] = duplicate
        item["ranking"] = {"rawScore": round(raw_score, 3), "adjustedScore": round(adjusted, 3)}
        return item


def build_review_prompt(reel: dict[str, Any], transcript_review: dict[str, Any], metrics: dict[str, Any]) -> str:
    """Build the user prompt for one reel review request."""

    payload = {
        "reel": {"id": reel.get("id"), "duration": reel.get("duration"), "start": reel.get("start"), "end": reel.get("end")},
        "transcriptReview": transcript_review,
        "activityMetrics": metrics,
    }
    return "Review this Instagram Reel candidate. Representative frames are attached chronologically.\n" + json.dumps(payload)


def parse_ollama_review_response(payload: dict[str, Any]) -> AIReviewResult:
    """Parse and validate an Ollama chat response as an AI review result."""

    content = ((payload.get("message") or {}).get("content") or payload.get("response") or "").strip()
    return AIReviewResult.model_validate_json(content)


def raise_for_ollama_status(response: httpx.Response) -> None:
    """Raise an HTTP error that includes Ollama's response body."""

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        body = response.text.strip()
        raise httpx.HTTPStatusError(
            f"{exc}; Ollama response body: {body}",
            request=exc.request,
            response=exc.response,
        ) from exc


def representative_timestamps(duration: float, count: int) -> list[float]:
    """Return representative timestamps including first and final meaningful frames."""

    if count <= 0 or duration <= 0:
        return []
    ratios = [0.0, 0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 0.98]
    if count != len(ratios):
        ratios = [index / max(count - 1, 1) for index in range(count)]
        ratios[-1] = 0.98
    return [round(min(max(duration * ratio, 0.0), max(duration - 0.1, 0.0)), 3) for ratio in ratios[:count]]


def media_duration(probe: dict[str, Any]) -> float:
    """Extract media duration from FFprobe output."""

    return float((probe.get("format") or {}).get("duration") or 0.0)


def first_video_stream(probe: dict[str, Any]) -> dict[str, Any]:
    """Return the first video stream from FFprobe output."""

    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "video":
            return stream
    return {}


def fps_from_stream(stream: dict[str, Any]) -> float:
    """Parse an FFprobe frame-rate fraction."""

    value = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
    numerator, denominator = value.split("/")
    return round(float(numerator) / max(float(denominator), 1.0), 3)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 checksum for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def overlaps(start_a: float, end_a: float, start_b: float, end_b: float) -> bool:
    """Return whether two time ranges overlap."""

    return max(start_a, start_b) < min(end_a, end_b)


def simple_segment(segment: dict[str, Any]) -> dict[str, Any]:
    """Return the transcript segment fields needed for review."""

    return {"start": float(segment.get("start", 0.0)), "end": float(segment.get("end", 0.0)), "text": str(segment.get("text", "")).strip()}


def speech_coverage_ratio(segments: list[dict[str, Any]], duration: float) -> float:
    """Return approximate speech coverage within a clip."""

    if duration <= 0:
        return 0.0
    spoken = sum(max(0.0, float(segment["end"]) - float(segment["start"])) for segment in segments)
    return clamp01(spoken / duration)


def clamp01(value: float) -> float:
    """Clamp a number to the inclusive zero-to-one range."""

    return max(0.0, min(1.0, float(value)))


def weighted_score(scores: dict[str, Any]) -> float:
    """Calculate the deterministic weighted base score."""

    weights = {
        "uploadPotential": 0.25,
        "gameplay": 0.20,
        "hook": 0.15,
        "excitement": 0.15,
        "payoff": 0.10,
        "contextIndependence": 0.05,
        "pacing": 0.05,
        "technicalQuality": 0.05,
    }
    return sum(float(scores.get(key, 0.0)) * weight for key, weight in weights.items())


def activity_penalty(reel: dict[str, Any]) -> float:
    """Return penalties for technically weak activity metrics."""

    visual = ((reel.get("activityMetrics") or {}).get("visual") or {})
    audio = ((reel.get("activityMetrics") or {}).get("audio") or {})
    penalty = 0.0
    penalty += float(visual.get("blackFrameRatio", 0.0) or 0.0) * 2.0
    penalty += float(visual.get("frozenFrameRatio", 0.0) or 0.0) * 1.5
    if float(visual.get("motionScore", 1.0) or 0.0) < 0.2 and float(audio.get("speechCoverage", 0.0) or 0.0) < 0.2:
        penalty += 1.0
    return penalty


def confidence_penalty(reel: dict[str, Any]) -> float:
    """Return a small penalty for low-confidence model judgments."""

    confidence = float((reel.get("aiReview") or {}).get("confidence", 1.0) or 0.0)
    return max(0.0, 0.6 - confidence)


def duplicate_analysis(reel: dict[str, Any], reels: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    """Return overlap-based duplicate analysis for a reel."""

    related: list[str] = []
    max_overlap = 0.0
    for other in reels:
        if other.get("id") == reel.get("id"):
            continue
        ratio = overlap_ratio(float(reel.get("start", 0.0)), float(reel.get("end", 0.0)), float(other.get("start", 0.0)), float(other.get("end", 0.0)))
        if ratio > max_overlap:
            max_overlap = ratio
        if ratio >= threshold:
            related.append(str(other.get("id")))
    return {"isPossibleDuplicate": bool(related), "relatedReelIds": related, "overlapRatio": round(max_overlap, 3), "penalty": 0.0}


def overlap_ratio(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    """Return overlap length divided by the shorter clip length."""

    overlap = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    shortest = max(min(end_a - start_a, end_b - start_b), 0.001)
    return overlap / shortest


def ranking_sort_key(reel: dict[str, Any]) -> tuple[float, float, float, float]:
    """Return the deterministic sort key for ranked reels."""

    ai_review = reel.get("aiReview") or {}
    scores = ai_review.get("scores") or {}
    return (
        -float((reel.get("ranking") or {}).get("adjustedScore", 0.0)),
        -float(ai_review.get("confidence", 0.0)),
        -float(scores.get("uploadPotential", 0.0)),
        float(reel.get("start", 0.0)),
    )


def ai_review_summary(reels: list[dict[str, Any]], model: str) -> dict[str, Any]:
    """Return video-level AI review stage summary from reel states."""

    completed = [reel for reel in reels if (reel.get("aiReview") or {}).get("status") == "COMPLETED"]
    failed = [reel for reel in reels if (reel.get("aiReview") or {}).get("status") == "FAILED"]
    recommendations = [(reel.get("aiReview") or {}).get("recommendation") for reel in completed]
    return {
        "status": "COMPLETED" if len(completed) == len(reels) and not failed else "PENDING",
        "completed": len(completed) == len(reels) and not failed,
        "model": model,
        "totalReels": len(reels),
        "reviewedReels": len(completed),
        "failedReels": len(failed),
        "uploadRecommendations": recommendations.count("UPLOAD"),
        "manualReviewRecommendations": recommendations.count("REVIEW"),
        "skipRecommendations": recommendations.count("SKIP"),
        "completedAt": utc_iso(),
        "error": None,
    }


def review_reels(video: VideoDocument) -> dict[str, int]:
    """Return a compatibility summary for generated reels awaiting model scoring."""

    generated = video.stages.get("reels")
    total = int(generated.metadata.get("totalGenerated", 0)) if generated else 0
    return {"reviewed": 0, "approved": 0, "rejected": 0, "pendingRanking": total}
